import os
import tensorflow as tf
from model import MyModel
import numpy as np
import imghdr
from tqdm import tqdm
from multiprocessing import cpu_count
import glob

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_dir", "data/Public",
                       """Path to the data directory""")
tf.flags.DEFINE_string("checkpoint_dir", 'models/0_inception_resnet_v2',
                       """Path to checkpoint folder""")

tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch Size (default: 32)""")
tf.flags.DEFINE_integer("num_threads", 8,
                        """Number of threads for data processing (default: 2)""")

tf.flags.DEFINE_integer('image_size', 299, 'Train image size')

tf.flags.DEFINE_integer('tta_mode', 0, 'Test time augmentation mode')

tf.flags.DEFINE_string("net",
                       'inception_resnet_v2',
                       "[resnet_v2_{50,101,152,200}, inception_{v4,resnet_v2}]")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

FLAGS = tf.flags.FLAGS


def is_valid(img_path):
  if os.path.getsize(img_path) == 0:  # zero-byte files
    return False
  if imghdr.what(img_path) not in ['jpeg', 'png', 'gif']:  # invalid image files
    return False
  return True


def list_files():
  img_paths = []
  fns = []
  corrupted_fns = []
  for fn in tqdm(tf.gfile.ListDirectory(FLAGS.data_dir), 'Data Loading'):
    img_path = os.path.join(FLAGS.data_dir, fn)
    if is_valid(img_path):
      img_paths.append(img_path)
      fns.append(str(fn).split('.')[0])
    else:
      corrupted_fns.append(str(fn).split('.')[0])
  return img_paths, fns, corrupted_fns


def _resize_image(image, size):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
  image = tf.squeeze(image, [0])
  return image

def _crop_and_resize(image, box, size):
  image = tf.expand_dims(image, 0)
  image = tf.image.crop_and_resize(image, [box], [0], [size, size])
  image = tf.squeeze(image, [0])
  return image


def _parse_function(filename, label):
  img_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(img_string, channels=3)
  image = tf.to_float(image) / 255.0

  if FLAGS.tta_mode == 0:
    print('TTA = central cropping')
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = _resize_image(image, FLAGS.image_size)

  elif FLAGS.tta_mode == 1:
    print('TTA = horizontal flipping -> central cropping')
    image = tf.image.flip_left_right(image)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = _resize_image(image, FLAGS.image_size)

  elif FLAGS.tta_mode == 2:
    print('TTA = top-left cropping')
    image = _crop_and_resize(image, [0, 0, 0.8, 0.8], FLAGS.image_size)

  elif FLAGS.tta_mode == 3:
    print('TTA = top-right cropping')
    image = _crop_and_resize(image, [0, 0.2, 0.8, 1], FLAGS.image_size)

  elif FLAGS.tta_mode == 4:
    print('TTA = bottom-left cropping')
    image = _crop_and_resize(image, [0.2, 0, 1, 0.8], FLAGS.image_size)

  elif FLAGS.tta_mode == 5:
    print('TTA = bottom-right cropping')
    image = _crop_and_resize(image, [0.2, 0.2, 1, 1], FLAGS.image_size)

  elif FLAGS.tta_mode == 6:
    print('TTA = horizontal flipping -> top-left cropping')
    image = tf.image.flip_left_right(image)
    image = _crop_and_resize(image, [0, 0, 0.8, 0.8], FLAGS.image_size)

  elif FLAGS.tta_mode == 7:
    print('TTA = horizontal flipping -> top-right cropping')
    image = tf.image.flip_left_right(image)
    image = _crop_and_resize(image, [0, 0.2, 0.8, 1], FLAGS.image_size)

  elif FLAGS.tta_mode == 8:
    print('TTA = horizontal flipping -> bottom-left cropping')
    image = tf.image.flip_left_right(image)
    image = _crop_and_resize(image, [0.2, 0, 1, 0.8], FLAGS.image_size)

  elif FLAGS.tta_mode == 9:
    print('TTA = horizontal flipping -> bottom-right cropping')
    image = tf.image.flip_left_right(image)
    image = _crop_and_resize(image, [0.2, 0.2, 1, 1], FLAGS.image_size)

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image, label


def init_data_generator(img_paths, fns):
  with tf.device('/cpu:0'):
    num_batches = int(np.ceil(len(img_paths) / FLAGS.batch_size))

    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    fns = tf.convert_to_tensor(fns, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, fns))
    dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.batch_size)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    init_data = iterator.make_initializer(dataset)
    get_next = iterator.get_next()

    return init_data, get_next, num_batches


def init_model(features):
  model = MyModel(FLAGS.net)
  logits, end_points = model(features, tf.constant(False, tf.bool))

  predictions = {
    'classes': tf.argmax(logits, axis=1),
    'top_3': tf.nn.top_k(logits, k=3)[1],
    'probs': tf.nn.softmax(logits)
  }

  return predictions


def main(_):
  # Construct data generator
  img_paths, fns, corrupted_fns = list_files()

  # Build Graph
  x = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
  predictions = init_model(features=x)

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print('Loaded model from %s' % checkpoint_dir)

    for FLAGS.tta_mode in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
      init_data, get_next, num_batches = init_data_generator(img_paths, fns)
      sess.run(init_data)

      submit_file = open('submissions/{}_vote_{}_tta{}.csv'.format(
        len(glob.glob("submissions/*.csv")), FLAGS.net, FLAGS.tta_mode), 'w')
      submit_file.write('id,predicted\n')
      probs = {}

      loop = tqdm(range(num_batches), 'Predicting')
      for _ in loop:
        img_batch, fn_batch = sess.run(get_next)
        _top3_preds, _probs = sess.run([predictions['top_3'], predictions['probs']], feed_dict={x: img_batch})

        for fn, preds, prob in zip(fn_batch, _top3_preds, _probs):
          fn = fn.decode("utf-8")
          submit_file.write('{},{}\n'.format(fn, ' '.join([str(p) for p in preds.tolist()])))
          probs[fn] = prob

      for fn in corrupted_fns:
        submit_file.write('{},93 83 2\n'.format(fn))
        probs[fn] = np.zeros([103])

      np.save('submissions/{}_probs_{}_tta{}.npy'.format(
        len(glob.glob("submissions/*.npy")), FLAGS.net, FLAGS.tta_mode), probs)
      submit_file.close()


if __name__ == '__main__':
  tf.app.run()
