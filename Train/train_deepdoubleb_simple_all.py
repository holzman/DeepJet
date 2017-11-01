import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
#import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from training_base import training_base
from Losses import loss_NLL
import sys
from DataCollection import DataCollection

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''

args = MyClass()
args.inputDataCollection = ' ../convertFromRoot/convert_20170717_ak8_deepDoubleB_init_train_val_fixQCD/dataCollection.dc'
args.outputDir = 'train_simple_one_file'

if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(args.outputDir)
    
# To use the full data:
# traind=DataCollection()
# traind.readFromFile(args.inputDataCollection)
# NENT = 1
# features_val = [fval[::NENT] for fval in traind.getAllFeatures()]
# labels_val=traind.getAllLabels()[0][::NENT,:]
##weights_val=traind.getAllWeights()[0][::NENT]
##spectators_val = traind.getAllSpectators()[0][::NENT,0,:]

# To use one data file:
import h5py
h5File = h5py.File('../convertFromRoot/convert_20170717_ak8_deepDoubleB_init_train_val_fixQCD/ntuple_merged_1.z')
features_val = [h5File["x0"][()], h5File["x1"][()]]
labels_val = h5File["y0"][()]
#weights_val = h5File["w0"][()]
#spectators_val = h5File["z0"][()]

print features_val[0].shape
print features_val[1].shape
print labels_val.shape

#X_train_val = features_val[0][:,0,:]
#y_train_val = labels_val[:,1]
#print X_train_val.shape
#print y_train_val.shape

from keras.optimizers import Adam, Nadam
from keras.layers import Input, concatenate, Flatten, Dense, Dropout
from keras.models import Model

def deep_model_doubleb_sv(inputs, num_classes):
    input_db = inputs[0]
    input_sv = inputs[1]
    x = Flatten()(input_db)
    sv = Flatten()(input_sv)
    concat = concatenate([x, sv], name='concat')
    fc = Dense(64, activation='relu',name='fc1_relu', kernel_initializer='lecun_uniform')(concat)
    fc = Dropout(rate=0.1, name='fc1_dropout')(fc)
    fc = Dense(32, activation='relu',name='fc2_relu', kernel_initializer='lecun_uniform')(concat)
    fc = Dropout(rate=0.1, name='fc2_dropout')(fc)
    fc = Dense(32, activation='relu',name='fc3_relu', kernel_initializer='lecun_uniform')(concat)
    fc = Dropout(rate=0.1, name='fc3_dropout')(fc)
    output = Dense(num_classes, activation='softmax', name='softmax', kernel_initializer='lecun_uniform')(fc)
    model = Model(inputs=inputs, outputs=output)
    print model.summary()
    return model


keras_model = deep_model_doubleb_sv([Input(shape=(1,27)),Input(shape=(5,14))], 2)

startlearningrate=0.0001
adam = Adam(lr=startlearningrate)
keras_model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

from DeepJet_callbacks import DeepJet_callbacks
        
callbacks=DeepJet_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=args.outputDir)

keras_model.fit(features_val, labels_val, batch_size = 1024, epochs = 1000,
                validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
