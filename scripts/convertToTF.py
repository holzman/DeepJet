#!/usr/bin/env python

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
            
            
from keras.models import load_model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os
import tensorflow as tf

parser = ArgumentParser('')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')


model=load_model(args.inputModel, custom_objects=global_loss_list)



num_output = 2


pred = [None]*num_output
pred_node_names = [None]*num_output
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'constant_graph_weights.pb'
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

K.set_learning_phase(0)
tfsession=K.get_session()
saver = tf.train.Saver()
tfoutpath=args.outputDir+'/tf.checkpoint'


#os.system('mkdir -p '+tfoutpath)

f = 'only_the_graph_def.pb.ascii'
tf.train.write_graph(tfsession.graph.as_graph_def(), args.outputDir, f, as_text=True)
print('saved the graph definition in ascii format at: ', os.path.join(args.outputDir, f))

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(tfsession, tfsession.graph.as_graph_def(), pred_node_names)


graph_io.write_graph(constant_graph, args.outputDir, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', os.path.join(args.outputDir, output_graph_name))

saver.save(tfsession, tfoutpath)





