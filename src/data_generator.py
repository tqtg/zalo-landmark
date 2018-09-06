import tensorflow as tf
import pandas as pd
import numpy as np
import collections

from preprocessing.inception_preprocessing import preprocess_image
from sklearn.utils import resample, shuffle

FLAGS = tf.flags.FLAGS


def _parse_function_train(filename, label):
  # # load and preprocess the image
  img_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(img_string, channels=3)
  image = tf.to_float(image) / 255.0
  image = preprocess_image(image, FLAGS.image_size, FLAGS.image_size, is_training=True, fast_mode=True)
  return image, label


def _parse_function_test(filename, label):
  img_string = tf.read_file(filename)
  image = tf.image.decode_jpeg(img_string, channels=3)
  image = tf.to_float(image) / 255.0
  image = preprocess_image(image, FLAGS.image_size, FLAGS.image_size)
  return image, label


def get_class_weights():
  with open('data/class_dis.txt', 'r') as f:
    id_weight_dic = {}
    for line in f:
      id, weight = line.strip().split('\t')
      id_weight_dic[int(id)] = float(weight)

    class_weights = []
    for _, weight in collections.OrderedDict(sorted(id_weight_dic.items())).items():
      class_weights.append(weight)

    class_weights = np.array(class_weights)
    class_weights = np.max(class_weights) / class_weights
    class_weights = np.log(class_weights)

    return class_weights


def _read_data_file(data_file, up_sample=False):
  """Read the content of the text file and store it into lists."""
  print('Loading data: {}'.format(data_file))
  df = pd.read_csv(data_file, header=None)
  img_paths = df[0].values
  labels = df[1].values

  # Data up sampling
  if up_sample:
    label_data = {}
    label_num_samples = {}
    for img_path, label in zip(img_paths, labels):
      if label not in label_data.keys():
        label_data[label] = []
      label_data[label].append(img_path)

    img_paths = []
    labels = []
    class_weights = get_class_weights()

    class_weights -= 2.0
    class_weights = np.where(class_weights > 0, class_weights, np.zeros_like(class_weights))

    for label, label_img_paths in label_data.items():
      img_paths += label_img_paths
      labels += [label] * len(label_img_paths)

      num_samples = int(class_weights[label] * len(label_img_paths))
      img_paths += resample(label_img_paths, n_samples=num_samples, replace=True)
      labels += [label] * num_samples
      label_num_samples[label] = num_samples

    _num_samples = []
    for _, n in collections.OrderedDict(sorted(label_num_samples.items())).items():
      _num_samples.append(n)

    print('Number of up-samples:\n{}'.format(np.array(_num_samples)))

  img_paths, labels = shuffle(img_paths, labels)
  return img_paths, labels


class DataGenerator(object):
  def __init__(self, train_file, test_file, batch_size, num_threads, buffer_size=10000, train_shuffle=True,
               up_sample=False, train_map_fn=None, test_map_fn=None):
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.buffer_size = buffer_size
    self.train_shuffle = train_shuffle
    self.train_map_fn = train_map_fn
    self.test_map_fn = test_map_fn

    # read datasets from csv files
    self.train_img_paths, self.train_labels = _read_data_file(train_file, up_sample=up_sample)
    self.test_img_paths, self.test_labels = _read_data_file(test_file)

    # number of batches per epoch
    self.train_batches_per_epoch = int(np.ceil(len(self.train_labels) / batch_size))
    self.test_batches_per_epoch = int(np.ceil(len(self.test_labels) / batch_size))

    # build datasets
    self._build_train_set()
    self._build_test_set()

    # create an reinitializable iterator given the dataset structure
    self.iterator = tf.data.Iterator.from_structure(self.train_set.output_types,
                                                    self.train_set.output_shapes)
    self.train_init_opt = self.iterator.make_initializer(self.train_set)
    self.test_init_opt = self.iterator.make_initializer(self.test_set)
    self.next = self.iterator.get_next()

  def load_train_set(self, session):
    session.run(self.train_init_opt)

  def load_test_set(self, session):
    session.run(self.test_init_opt)

  def get_next(self, session):
    return session.run(self.next)

  def _build_data_set(self, img_paths, labels, map_fn, shuffle=False):
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    data = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    if shuffle:
      data = data.shuffle(buffer_size=self.buffer_size)
    data = data.map(map_fn, num_parallel_calls=self.num_threads)
    data = data.batch(self.batch_size)
    data = data.prefetch(self.batch_size)
    return data

  def _build_train_set(self):
    if self.train_map_fn == None:
      self.train_map_fn = _parse_function_train
    self.train_set = self._build_data_set(self.train_img_paths,
                                          self.train_labels,
                                          self.train_map_fn,
                                          self.train_shuffle)

  def _build_test_set(self):
    if self.test_map_fn == None:
      self.test_map_fn = _parse_function_test
    self.test_set = self._build_data_set(self.test_img_paths,
                                         self.test_labels,
                                         self.test_map_fn)


if __name__ == '__main__':
  import os

  # Prepare data
  train_file = os.path.join('data/train.csv')
  val_file = os.path.join('data/val.csv')
  generator = DataGenerator(train_file, val_file, 32, 4)
