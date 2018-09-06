import os
import tensorflow as tf
from tqdm import tqdm
from model import MyModel, _NUM_CLASSES, count_parameters
from data_generator import DataGenerator
import collections
import numpy as np

slim = tf.contrib.slim

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_dir", "data", "Path to the data directory")
tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints', "Path to checkpoint folder")
tf.flags.DEFINE_string("log_dir", 'log', "Path to log folder")

tf.flags.DEFINE_integer("num_threads", 8, "Number of threads for data processing")
tf.flags.DEFINE_integer("display_step", 20, "Display after number of steps")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size")

tf.flags.DEFINE_integer('image_size', 299, 'Train image size')

tf.flags.DEFINE_string("net",
                       'inception_resnet_v2',
                       # 'resnet_v2_152',
                       # 'resnet_v2_50',
                       # 'inception_v4',
                       # 'densenet161',
                       # 'pnasnet_large',
                       "[resnet_v2_{50,101,152,200}, inception_{v4,resnet_v2}]")

tf.flags.DEFINE_integer("train_mode", 2,
                        "[1: train from scratch, 2: from pre-trained, 3: re-train from trained model]")

tf.flags.DEFINE_integer("fold", -1,
                        "-1 or [0,..,9]")

tf.flags.DEFINE_string('optimizer',
                       'momentum',
                       'The name of the optimizer, one of "adam", "rmsprop", or "momentum".')

tf.flags.DEFINE_float("learning_rate", 0.01,
                      "Initial learning rate")

tf.flags.DEFINE_string('lr_decay_rule',
                       'step',
                       # 'kar',
                       'Specifies how the learning rate is decayed. One of "fixed", "exp", or "step"')

tf.flags.DEFINE_string("loss_weighted",
                       'linear',
                       "[none, linear, ln, log2, log10]")
tf.flags.DEFINE_float('weight_decay', 0.00004, "The weight decay on the model weights")

#####################
# Fine-Tuning Flags #
#####################

tf.flags.DEFINE_string(
  'pretrained_dir', 'pretrained',
  'The path to a checkpoint from which to fine-tune.')

tf.flags.DEFINE_string(
  'checkpoint_exclude_scopes',

  # None,
  'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits',
  # 'InceptionV4/Logits,InceptionV4/AuxLogits',
  # 'resnet_v2_152/logits',
  # 'resnet_v2_50/logits',
  # 'densenet161/logits',
  # 'final_layer/FC,aux_7/aux_logits',

  'Comma-separated list of scopes of variables to exclude when restoring '
  'from a checkpoint.')

tf.flags.DEFINE_string(
  'trainable_scopes',

  None,

  'Comma-separated list of scopes to filter the set of variables to train.'
  'By default, None would train all the variables.')

tf.flags.DEFINE_boolean(
  'ignore_missing_vars', False,
  'When restoring a checkpoint would ignore missing variables.')

####################
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")

FLAGS = tf.flags.FLAGS


def init_data_generator():
  # Prepare data
  if FLAGS.fold == -1:
    train_file = os.path.join(FLAGS.data_dir, 'train.csv'.format(FLAGS.fold))
    test_file = os.path.join(FLAGS.data_dir, 'test.csv'.format(FLAGS.fold))
  else:
    train_file = os.path.join(FLAGS.data_dir, 'fold{}_train.csv'.format(FLAGS.fold))
    test_file = os.path.join(FLAGS.data_dir, 'fold{}_test.csv'.format(FLAGS.fold))

  # Place data loading and preprocessing on the cpu
  with tf.device('/cpu:0'):
    generator = DataGenerator(train_file, test_file, FLAGS.batch_size, FLAGS.num_threads)
  return generator


def learning_rate_with_decay(batches_per_epoch, boundary_epochs, decay_rates):
  initial_learning_rate = FLAGS.learning_rate

  # Reduce the learning rate at certain epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def get_class_weights():
  if FLAGS.loss_weighted == 'none':
    return tf.constant(1.0, dtype=tf.float32)

  with open(os.path.join(FLAGS.data_dir, 'class_dis.txt'), 'r') as f:
    id_weight_dic = {}
    for line in f:
      id, weight = line.strip().split('\t')
      id_weight_dic[int(id)] = float(weight)

    class_weights = []
    for _, weight in collections.OrderedDict(sorted(id_weight_dic.items())).items():
      class_weights.append(weight)

    class_weights = np.array(class_weights)
    class_weights = np.max(class_weights) / class_weights  # linear

    if FLAGS.loss_weighted == 'ln':
      class_weights = 1.0 + np.log(class_weights)
    elif FLAGS.loss_weighted == 'log2':
      class_weights = 1.0 + np.log2(class_weights)
    elif FLAGS.loss_weighted == 'log10':
      class_weights = 1.0 + np.log10(class_weights)

    class_weights = class_weights / class_weights.sum() * _NUM_CLASSES
    print('Loss weights {}:\n{}'.format(FLAGS.loss_weighted, class_weights))

    return tf.constant(class_weights, dtype=tf.float32)


def _get_variables_to_train(trainable_scopes):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def build_model(features, labels, training, train_batches):
  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  model = MyModel(FLAGS.net, weight_decay=FLAGS.weight_decay)
  logits, end_points = model(features, training)

  predictions = {
    'classes': tf.argmax(logits, axis=1),
    'top_3': tf.nn.top_k(logits, k=3)[1]
  }

  onehot_labels = tf.one_hot(labels, depth=_NUM_CLASSES)
  loss_weights = tf.cond(training,
                         lambda: tf.reduce_sum(onehot_labels * get_class_weights(), axis=1, keepdims=True),
                         lambda: tf.constant(1.0, dtype=tf.float32))
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(loss_weights * cross_entropy)

  if 'AuxLogits' in end_points:
    aux_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=end_points['AuxLogits'])
    loss += 0.4 * tf.reduce_mean(loss_weights * aux_cross_entropy)

  # Create a tensor named cross_entropy for logging purposes.
  tf.summary.scalar('optimization/loss', loss)

  global_step = tf.train.get_or_create_global_step()

  if FLAGS.lr_decay_rule == 'exp':
    learning_rate = tf.train.exponential_decay(
      learning_rate=FLAGS.learning_rate,
      global_step=global_step,
      decay_steps=train_batches * 10,
      decay_rate=0.5,
      staircase=True
    )
  elif FLAGS.lr_decay_rule == 'step':
    learning_rate = learning_rate_with_decay(batches_per_epoch=train_batches,
                                             boundary_epochs=[8, 20, 30, 40],
                                             decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])(global_step)
  elif FLAGS.lr_decay_rule == 'kar':
    current_epoch = tf.ceil(global_step / train_batches)
    learning_rate = tf.cond(current_epoch <= 2,
                            lambda: FLAGS.learning_rate,
                            lambda: FLAGS.learning_rate * tf.pow(0.8, tf.to_float(current_epoch - 2)))
  else:
    learning_rate = FLAGS.learning_rate

  tf.summary.scalar('optimization/learning_rate', learning_rate)

  if FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1.0)
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  ################## Update all params #####################
  variables_to_train = _get_variables_to_train(FLAGS.trainable_scopes)
  count_parameters(variables_to_train)

  gradients = tf.gradients(loss, variables_to_train)
  grad_updates = optimizer.apply_gradients(zip(gradients, variables_to_train),
                                           global_step=global_step, name='train_op')
  train_op = tf.group(grad_updates, update_ops)

  ################## Evaluation ###########################
  top_1_acc, update_top_1 = tf.metrics.accuracy(labels, predictions['classes'], name='metrics')
  top_3_acc, update_top_3 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=labels, k=3), name='metrics')

  running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init = tf.variables_initializer(var_list=running_vars)
  metrics_update = tf.group([update_top_1, update_top_3])

  top_1_error = 1.0 - top_1_acc
  top_3_error = 1.0 - top_3_acc

  metrics = {'init': metrics_init,
             'update': metrics_update,
             'top_1_error': top_1_error,
             'top_3_error': top_3_error}

  tf.summary.scalar('metrics/top_1_error', top_1_error)
  tf.summary.scalar('metrics/top_3_error', top_3_error)

  return train_op, loss, predictions, metrics


def init_model(sess, checkpoint_dir):
  sess.run(tf.global_variables_initializer())

  if FLAGS.train_mode == 2:
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
      exclusions = [scope.strip()
                    for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)

    checkpoint_path = os.path.join(FLAGS.pretrained_dir, '{}/{}.ckpt'.format(FLAGS.net, FLAGS.net))
    print('Weights initialized from %s' % checkpoint_path)

    slim.assign_from_checkpoint_fn(checkpoint_path,
                                   variables_to_restore,
                                   ignore_missing_vars=FLAGS.ignore_missing_vars)(sess)

  elif FLAGS.train_mode == 3:
    checkpoint_path = checkpoint_dir.replace('train_mode=3', 'train_mode=2')
    print('Weighted loaded from %s' % checkpoint_path)
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(checkpoint_path))


def main(_):
  model_name = '{}_fold{}'.format(FLAGS.net, FLAGS.fold)

  # Construct data generator
  generator = init_data_generator()

  # Build Graph
  x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
  y = tf.placeholder(tf.int32, shape=[None])
  training = tf.placeholder(tf.bool)

  train_op, loss, predictions, metrics = build_model(features=x, labels=y, training=training,
                                                     train_batches=generator.train_batches_per_epoch)

  # Summary for monitoring
  summary = tf.summary.merge_all()

  # Create writers for logging and visualization
  log_dir = os.path.join(FLAGS.log_dir, model_name)
  if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
  tf.gfile.MakeDirs(log_dir)

  train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
  test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
  if tf.gfile.Exists(checkpoint_dir):
    tf.gfile.DeleteRecursively(checkpoint_dir)
  tf.gfile.MakeDirs(checkpoint_dir)

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    init_model(sess, checkpoint_dir)

    ##############################################################
    best_error = float('inf')
    best_epoch = 0

    print("\nStart training...")
    print("Open Tensorboard at --logdir {}".format(log_dir))

    # Loop over number of epochs
    for epoch in range(FLAGS.num_epochs):
      print("\nEpoch: {}/{}".format(epoch + 1, FLAGS.num_epochs))

      # Training
      sess.run(metrics['init'])
      sum_loss = 0.0
      num_examples = 0
      generator.load_train_set(sess)
      loop = tqdm(range(generator.train_batches_per_epoch), 'Training')
      for step in loop:
        # Get next batch of data
        img_batch, label_batch = generator.get_next(sess)

        _, _loss, _ = sess.run([train_op, loss, metrics['update']],
                               feed_dict={x: img_batch,
                                          y: label_batch,
                                          training: True})

        loop.set_postfix(loss=_loss)
        num_examples += len(label_batch)
        sum_loss += _loss * len(label_batch)

        # Generate summary with the current batch of data and write to file
        if step % (FLAGS.display_step * 5) == 0:
          s = sess.run(summary, feed_dict={x: img_batch,
                                           y: label_batch,
                                           training: False})
          current_step = epoch * generator.train_batches_per_epoch + step
          train_writer.add_summary(s, current_step)

      print('train_loss = {:.4f}, top_1_error = {:.4f}, top_3_error = {:.4f}'.format(
        sum_loss / num_examples, sess.run(metrics['top_1_error']), sess.run(metrics['top_3_error'])))

      ##################################################################
      # Evaluation
      sess.run(metrics['init'])
      # sum_loss = 0.0
      # num_examples = 0
      generator.load_test_set(sess)
      loop = tqdm(range(generator.test_batches_per_epoch), 'Testing')
      for step in loop:
        # Get next batch of data
        img_batch, label_batch = generator.get_next(sess)
        # And run the training op, get loss and accuracy
        _, _loss, _summary = sess.run([metrics['update'], loss, summary],
                                      feed_dict={x: img_batch,
                                                 y: label_batch,
                                                 training: False})
        loop.set_postfix(loss=_loss)
        # num_examples += len(label_batch)
        # sum_loss += _loss * len(label_batch)
        current_step = epoch * generator.test_batches_per_epoch + step

        if step % FLAGS.display_step == 0:
          test_writer.add_summary(_summary, current_step)

      # val_loss = sum_loss / num_examples
      # print('val_loss = {:.4f}, top_1_error = {:.4f}, top_3_error = {:.4f}'.format(
      #   val_loss, sess.run(metrics['top_1_error']), sess.run(metrics['top_3_error'])))

      top1_error = sess.run(metrics['top_1_error'])
      top3_error = sess.run(metrics['top_3_error'])
      print('top_1_error = {:.4f}, top_3_error = {:.4f}'.format(top1_error, top3_error))

      ###################################################################
      if best_error >= top1_error:
        best_error = top1_error
        best_epoch = epoch + 1
        checkpoint_prefix = os.path.join(checkpoint_dir, 'epoch_{}'.format(epoch + 1))
        path = saver.save(sess, checkpoint_prefix)
        print("Saved model checkpoint to {}".format(path))

      print('Best top_1_error = {:.4f} @ epoch = {}'.format(best_error, best_epoch))

    print('Done')


if __name__ == '__main__':
  tf.app.run()
  # get_class_weights()
