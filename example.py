import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_xlogx import xlogx as tf_xlogx

xs = np.linspace(0,3,100)
sess = tf.Session()
ys = sess.run(tf_xlogx(xs))
plt.plot(xs,ys)
plt.show()
