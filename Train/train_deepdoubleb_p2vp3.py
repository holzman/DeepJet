
# coding: utf-8

# In[1]:

import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')


# In[2]:

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''


# In[5]:
import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

args = MyClass()
args.inputDataCollection = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_pf_cpf_sv_train_val/dataCollection.dc'
args.outputDir = 'train_deep_full_5x100_b1024_e20'

#also does all the parsing
train=training_base(testrun=False,args=args)


if not train.modelSet():
    from DeepJet_models_ResNet import deep_model_full,conv_model_full

    train.setModel(deep_model_full)
    #train.setModel(conv_model_full)
    
    train.compileModel(learningrate=0.0001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])
    

model,history,callbacks = train.trainModel(nepochs=20, 
                                batchsize=1024, 
                                 stop_patience=1000, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0000001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0000001, 
                                           maxqsize=100)




