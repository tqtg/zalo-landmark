import os
import tensorflow as tf
from model import MyModel
from data_generator import DataGenerator
from tqdm import tqdm

from preprocessing.inception_preprocessing import preprocess_image

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_dir", "data",
                       """Path to the data directory""")
tf.flags.DEFINE_string("checkpoint_dir",
                       'models/0_inception_resnet_v2',
                       """Path to checkpoint folder""")

tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch Size (default: 32)""")
tf.flags.DEFINE_integer("num_threads", 8,
                        """Number of threads for data processing (default: 2)""")

tf.flags.DEFINE_integer("fold", -1, "")

tf.flags.DEFINE_integer('image_size', 299, 'Train image size')

tf.flags.DEFINE_string("net",
                       'inception_resnet_v2',
                       # 'resnet_v2_152',
                       # 'inception_v4',
                       # 'densenet161',
                       "[resnet_v2_{50,101,152,200}, inception_{v4,resnet_v2}]")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

FLAGS = tf.flags.FLAGS


def map_fn(filename, label):
  img_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(img_string, channels=3)
  image = tf.to_float(image) / 255.0
  image = preprocess_image(image, FLAGS.image_size, FLAGS.image_size)
  return image, filename, label


def init_data_generator():
  if FLAGS.fold == -1:
    # Prepare data
    train_file = os.path.join(FLAGS.data_dir, 'train.csv')
    val_file = os.path.join(FLAGS.data_dir, 'val.csv')
  else:
    train_file = os.path.join(FLAGS.data_dir, 'fold{}_train.csv'.format(FLAGS.fold))
    val_file = os.path.join(FLAGS.data_dir, 'fold{}_test.csv'.format(FLAGS.fold))

  # Place data loading and preprocessing on the cpu
  with tf.device('/cpu:0'):
    generator = DataGenerator(train_file, val_file, FLAGS.batch_size, FLAGS.num_threads,
                              train_map_fn=map_fn, test_map_fn=map_fn)
  return generator


def init_model(features, labels):
  model = MyModel(FLAGS.net)
  logits, end_points = model(features, tf.constant(False, dtype=tf.bool))

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(cross_entropy)

  predictions = {
    'classes': tf.argmax(logits, axis=1),
    'top_3': tf.nn.top_k(logits, k=3)[1]
  }

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

  return loss, predictions, metrics


def main(_):
  # Construct data generator
  generator = init_data_generator()

  # Build Graph
  x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
  y = tf.placeholder(tf.int32, shape=[None])
  loss, predictions, metrics = init_model(features=x, labels=y)

  submit_file = open('eval/{}_{}_fold{}.csv'.format(len(os.listdir('eval/')), FLAGS.net, FLAGS.fold), 'w')
  # submit_file.write('id,predicted\n')

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print('Loaded model from %s' % checkpoint_dir)

    ##################################################################
    # Evaluatin
    sess.run(metrics['init'])
    sum_loss = 0.0
    num_examples = 0
    labels = []
    preds = []
    generator.load_test_set(sess)
    loop = tqdm(range(generator.test_batches_per_epoch), 'Testing')
    for _ in loop:
      # Get next batch of data
      img_batch, fn_batch, label_batch = generator.get_next(sess)

      # And run the training op, get loss and accuracy
      _, _loss, _preds, _top3_preds = sess.run([metrics['update'], loss, predictions['classes'], predictions['top_3']],
                                               feed_dict={x: img_batch,
                                                          y: label_batch})

      # for fn, _top3_pred in zip(fn_batch, _top3_preds):
      #   fn = fn.decode("utf-8").split('/')[-1].split('.')[0]
      #   submit_file.write('{},{}\n'.format(fn, ' '.join([str(p) for p in _top3_pred.tolist()])))

      num_examples += len(label_batch)
      sum_loss += _loss * len(label_batch)

      preds.extend(_preds.tolist())
      labels.extend(label_batch.tolist())

    val_loss = sum_loss / num_examples
    print('val_loss = {:.4f}, top_1_error = {:.4f}, top_3_error = {:.4f}'.format(
      val_loss, sess.run(metrics['top_1_error']), sess.run(metrics['top_3_error'])))

    submit_file.write('\nval_loss = {:.4f}, top_1_error = {:.4f}, top_3_error = {:.4f}'.format(
      val_loss, sess.run(metrics['top_1_error']), sess.run(metrics['top_3_error'])))



    # y_ = tf.placeholder(tf.int32, [None])
    # y = tf.placeholder(tf.int32, [None])
    # confusion = sess.run(tf.confusion_matrix(labels=y, predictions=y_),
    #                      feed_dict={y_: np.array(preds), y: np.array(labels)})
    # print('Confusion matrix:\n{}'.format(confusion))
    # np.savetxt('confusion.txt', confusion, fmt='%d', delimiter='\t')
    #
    # correct = confusion * np.identity(np.shape(confusion)[0])
    # print('Correct:\n{}'.format(correct.diagonal()))
    #
    # sum = confusion.dot(np.ones(np.shape(confusion)[0]))
    # print('Sum:\n{}'.format(sum))
    #
    # acc = correct / sum
    # print('Acc:\n{}'.format(acc.diagonal()))


if __name__ == '__main__':
  tf.app.run()
