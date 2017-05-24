import numpy as np
import tensorflow as tf

## Tensor

scalar = tf.constant([1])  # <tf.Tensor 'Const:0' shape=(1,) dtype=int32>
# Shape: TensorShape([Dimension(1)]) 
vector = tf.constant(np.arange(1,5))  # <tf.Tensor 'Const_1:0' shape=(4,) dtype=int64>
# Shape: TensorShape([Dimension(4)]) 
matrix = tf.constant(np.arange(1,16).reshape(3,5))  # <tf.Tensor 'Const_2:0' shape=(3, 5) dtype=int64>
# Shape: TensorShape([Dimension(3), Dimension(5)])
tensor = tf.constant([[ [1,2,3], [2,3,4], [3,4,5] ],
                      [ [4,5,6], [5,6,7], [6,7,8] ],
                      [ [7,8,9], [8,9,10], [9,10,11]]
                     ])  # <tf.Tensor 'Const_3:0' shape=(3, 3, 3) dtype=int32>
# Shape: TensorShape([Dimension(3), Dimension(3), Dimension(3)])
# import ipdb; ipdb.set_trace()
# Show tensor's values
with tf.Session() as session:
    print "Print Scalar:"
    res_scalar = session.run(scalar)
    print res_scalar, ",its shape: ", res_scalar.shape
    print "Print Vector:"
    res_vector = session.run(vector)
    print res_vector, ",its shape: ", res_vector.shape
    print "Print Matrix 3*5:"
    res_matrix = session.run(matrix)
    print res_matrix, ",its shape: ", res_matrix.shape
    print "Print Tensor 3*3*3:"
    res_tensor = session.run(tensor)
    print res_tensor, ",its shape: ", res_tensor.shape
    
# Multiply operation
multiplier = tf.constant([ [1,2,3], [4,5,6] ])
multiplicand  = tf.constant([ [2,2], [2,2], [2,2] ])
multiplier2 = tf.constant([ [2,2,2], [2,2,2] ])
product = tf.matmul(multiplier, multiplicand)
ele_wise_prod = multiplier * multiplier2
with tf.Session() as mul_session:
    res = mul_session.run(product)
    print res, ",its shape: ", res.shape
    res = mul_session.run(ele_wise_prod)
    print res, " its shape: ", res.shape

## Variable
variable = tf.Variable(0)
print variable
one_const = tf.constant(1)
print one_const," its shape: " , one_const.shape
new_var = tf.add(variable, one_const)
print new_var
update_var = tf.assign(variable, new_var)

# Variables need to be initialized before a graph can be run in a session
init_op = tf.global_variables_initializer()

with tf.Session() as init_session:
    init_session.run(init_op)
    res = init_session.run(variable)
    print res
    for _ in range(3):
        res = init_session.run(update_var)
        print res
        
# Placeholder
