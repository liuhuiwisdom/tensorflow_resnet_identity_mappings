import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _kernel_weights(shape, stddev, name, w=1):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    weight_decay = w*tf.nn.l2_loss(var, name='weight_loss')
    tf.add_to_collection('l2_losses', weight_decay)
    return var


def _bias(shape, name):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.constant_initializer(0., dtype=dtype))
    return var

def leaky_relu(x, alpha=0.1, max_value=None):

    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))

    x -= tf.constant(alpha, dtype=tf.float32) * negative_part

    return x


def weight_variable(shape, w=1, name=None):
    return _kernel_weights(shape, w=w, stddev=0.1, name=name)


def bias_variable(shape, name=None):
    return _bias(shape, name=name)


def softmax_layer(inpt, shape, name=''):
    fc_w = weight_variable(shape, name=name+'_weights')
    fc_b = bias_variable([shape[1]], name=name+'_bias')

    logits = tf.matmul(inpt, fc_w) + fc_b
    activation = tf.nn.softmax(logits)

    return activation, logits


def conv_layer(inpt, filter_shape, stride, name=''):
    filter_ = weight_variable(filter_shape, name=name+'_weights')
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    normalized = batch_norm(conv, name=name)
    out = tf.nn.relu(normalized)
    return out


def conv_layer_resnet_im(inpt, filter_shape, stride, name=''):
    filter_ = weight_variable(filter_shape, name=name + '_weights')

    normalized = batch_norm(inpt, name=name)

    activated = tf.nn.relu(normalized)

    conv = tf.nn.conv2d(activated, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    return conv


def batch_norm(inpt, name=''):
    channels = inpt.get_shape().as_list()[3]
    mean, var = tf.nn.moments(inpt, axes=[0, 1, 2])
    beta = bias_variable([channels], name=name + '_beta')
    gamma = weight_variable([channels], w=0, name=name + '_gamma')

    inpt_normalized = tf.nn.batch_norm_with_global_normalization(inpt, mean, var, beta, gamma, 0.001, scale_after_normalization=True)

    return inpt_normalized


def max_pool_layer(inpt, filter, stride):
    filter_ = [1, filter, filter, 1]
    strides_ = [1, stride, stride, 1]
    outpt = tf.nn.max_pool(inpt, ksize=filter_, strides=strides_, padding='SAME')
    return outpt


def residual_block_deep_im(inpt, output_depth, down_sample, projection=False, name=''):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer_resnet_im(inpt, [1, 1, input_depth, int(output_depth // 4)], 1, name=name+'_conv1')
    conv2 = conv_layer_resnet_im(conv1, [3, 3, int(output_depth // 4), int(output_depth // 4)], 1, name=name+'_conv2')
    conv3 = conv_layer_resnet_im(conv2, [1, 1, int(output_depth // 4), output_depth], 1, name=name + '_conv3')

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2, name=name+'_conv_projection')
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv3 + input_layer

    return res


def residual_block_im(inpt, output_depth, down_sample, projection=False, name=''):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer_resnet_im(inpt, [3, 3, input_depth, output_depth], 1, name=name+'_conv1')
    conv2 = conv_layer_resnet_im(conv1, [3, 3, output_depth, output_depth], 1, name=name+'_conv2')

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2, name=name+'_conv_projection')
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer

    return res

