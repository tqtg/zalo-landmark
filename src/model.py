from nets import resnet_v2
from nets import inception
from nets import nasnet
from nets import pnasnet
from nets import densenet
import tensorflow as tf

slim = tf.contrib.slim

_NUM_CLASSES = 103

networks_map = {'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
                'nasnet_large': nasnet.build_nasnet_large,
                'pnasnet_large': pnasnet.build_pnasnet_large,
                'densenet161': densenet.densenet161,
                }

arg_scopes_map = {'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'nasnet_large': nasnet.nasnet_large_arg_scope,
                  'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
                  'densenet161': densenet.densenet_arg_scope,
                  }


class MyModel():

  def __init__(self, net, weight_decay=0.0):
    self.func = networks_map[net]
    self.arg_scope = arg_scopes_map[net](weight_decay=weight_decay)

  def __call__(self, inputs, training, **kwargs):
    with slim.arg_scope(self.arg_scope):
      return self.func(inputs, _NUM_CLASSES, is_training=training, **kwargs)


def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)



def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
