import tensorflow as tf
from tensorflow.python.framework import ops
from pkg_resources import resource_filename, Requirement

path_to_lib = resource_filename(__name__, 'libtensorflow-xlogx.so')
xlogx_module = tf.load_op_library(path_to_lib)


@ops.RegisterGradient("XLogXOp")
def _xlogx_grad(op, grad):
    x = op.inputs[0]
    return [xlogx_module.x_log_x_grad_op(x)]


def xlogx(x):
    return xlogx_module.x_log_x_op(x)