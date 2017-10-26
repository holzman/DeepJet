# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
import sys
from cStringIO import StringIO
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# ####                                                                                                                                     # setup the environment
backup = sys.stdout
sys.stdout = StringIO()     # capture output                                                                                                
print_tensors_in_checkpoint_file(file_name='tf.checkpoint', tensor_name='', all_tensors=False)
# ####                                                                                                                                      
out = sys.stdout.getvalue() # release output                                                                                               

sys.stdout.close()  # close the stream 
sys.stdout = backup # restore original stdout
tensor_list = out.split('\n')
tensor_list = [t for t in tensor_list if t !=''] # remove ''
print tensor_list

tf.reset_default_graph()
tensor_dict = {}
for tensor in tensor_list:
    name = tensor.split(' ')[0]
    my_type = tensor.split(' ')[1]    
    my_shape = tensor.split(' ')[2]
    print name, eval(my_shape)
    tensor_dict[name] = tf.get_variable(name, eval(my_shape))

saver =  tf.train.Saver()
# Use the saver object normally after that.
with tf.Session() as sess:
    saver.restore(sess,'tf.checkpoint')
    print("Model restored.")
    for (name, tensor) in tensor_dict.iteritems():
        print '%s : %s' %(name, tensor.eval())
