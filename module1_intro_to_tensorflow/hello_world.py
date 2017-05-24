import tensorflow as tf

# building your first graph
a = tf.constant([2])  # <tf.Tensor 'Const:0' shape=(1,) dtype=int32>
b = tf.constant([3])

c = tf.add(a, b) # c = a + b is another way
# <tf.Tensor 'Add:0' shape=(1,) dtype=int32>

session = tf.Session()  # <tensorflow.python.client.session.Session object at 0x7f2bbfcfb110>
res = session.run(c)  # array([5], dtype=int32), shape: (1,) 
import ipdb; ipdb.set_trace()
print res
session.close()  # remember close the session

with tf.Session() as wrap_session:
    print "Auto close session"
    print wrap_session.run(c)

