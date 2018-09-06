import os
import pickle
import random
import imghdr

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

random.seed(2018)


def get_fns_lbs(base_dir, json_file, pickle_fn, force=False):
  if os.path.isfile(pickle_fn) and not force:
    fns_lbs_file = pickle.load(open(pickle_fn, 'rb'))
    fns = fns_lbs_file['fns']
    lbs = fns_lbs_file['lbs']
    cnt = fns_lbs_file['cnt']
    return fns, lbs, cnt

  f = open(json_file, 'r')
  line = f.readlines()[0]  # only one line

  end = 0
  id_marker = '\\"id\\": '
  cate_marker = '\\"category\\": '
  cnt = 0
  fns = []  # list of all image filenames
  lbs = []  # list of all labels
  while True:
    start0 = line.find(id_marker, end)
    if start0 == -1: break
    start_id = start0 + len(id_marker)
    end_id = line.find(',', start_id)

    start0 = line.find(cate_marker, end_id)
    start_cate = start0 + len(cate_marker)
    end_cate = line.find('}', start_cate)

    end = end_cate
    cnt += 1
    cl = line[start_cate:end_cate]
    fn = base_dir + cl + '/' + line[start_id:end_id] + '.jpg'
    if os.path.getsize(fn) == 0:  # zero-byte files
      continue
    if imghdr.what(fn) not in ['jpeg', 'png', 'gif']: # invalid image files
      continue
    lbs.append(int(cl))
    fns.append(fn)

  fns_lbs_file = {'fns': fns, 'lbs': lbs, 'cnt': cnt}
  pickle.dump(fns_lbs_file, open(pickle_fn, 'wb'))
  print(os.path.isfile(pickle_fn))

  return fns, lbs, cnt


def split_data(fns, lbs, val_ratio=0.1):
  train_set = []
  val_set = []

  lb_fn_dic = {}
  for fn, lb in zip(fns, lbs):
    if lb not in lb_fn_dic:
      lb_fn_dic[lb] = []
    lb_fn_dic[lb].append((fn, lb))

  ##################
  dis_file = open('data/class_dis.txt', 'w')
  data_dis = {}
  for lb in lb_fn_dic:
    data_dis[lb] = len(lb_fn_dic[lb]) / len(fns) * 100.0
    random.shuffle(lb_fn_dic[lb])
    split_idx = int(val_ratio * len(lb_fn_dic[lb]))
    val_set += lb_fn_dic[lb][:split_idx]
    train_set += lb_fn_dic[lb][split_idx:]

  for lb, rate in sorted(data_dis.items(), key=lambda x: x[1]):
    dis_file.write('%d\t%.3f\n' % (lb, rate))
  print('Class distribution in class_dis.txt')

  random.shuffle(train_set)
  random.shuffle(val_set)

  with open('data/train.csv', 'w') as f:
    for fn, lb in train_set:
      f.write('{},{}\n'.format(fn, lb))
  with open('data/val.csv', 'w') as f:
    for fn, lb in val_set:
      f.write('{},{}\n'.format(fn, lb))

  return train_set, val_set


if __name__ == "__main__":
  json_file = 'data/train_val2018.json'
  data_dir = 'data/TrainVal/'
  fns_lbs_file = 'data/fns_lbs.pkl'

  print('Loading data')
  fns, lbs, cnt = get_fns_lbs(data_dir, json_file, fns_lbs_file)

  print('Total files in the original dataset: {}'.format(cnt))
  print('Total valid files: {}'.format(len(fns)))
  print('Total corrupted files {}'.format(cnt - len(fns)))

  print('Split data')

  fns, lbs = np.asarray(fns), np.asarray(lbs)
  fns, lbs = shuffle(fns, lbs, random_state=2018)

  kf = KFold(n_splits=10, random_state=2018)
  fold = 0
  for train_index, test_index in kf.split(fns):
    print("TRAIN:", train_index, "TEST:", test_index)
    fns_train, fns_test = fns[train_index], fns[test_index]
    lbs_train, lbs_test = lbs[train_index], lbs[test_index]

    with open('data/fold{}_train.csv'.format(fold), "w") as f:
      for fn, lb in zip(fns_train, lbs_train):
        f.write('{},{}\n'.format(fn, lb))

    with open('data/fold{}_test.csv'.format(fold), "w") as f:
      for fn, lb in zip(fns_test, lbs_test):
        f.write('{},{}\n'.format(fn, lb))

    fold += 1

  train_set, val_set = split_data(fns, lbs, val_ratio=0.08)
  print('Number of training images: {}'.format(len(train_set)))
  print('Number of validation images: {}'.format(len(val_set)))
