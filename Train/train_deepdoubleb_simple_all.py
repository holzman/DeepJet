import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
import setGPU
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
args.inputDataCollection = '/data/shared/BumbleB/convert_deepDoubleB_simple_all/dataCollection.dc'
args.outputDir = 'train_deep_simple_all'

#also does all the parsing
#train=training_base(testrun=False,args=args)

traind=DataCollection()
traind.readFromFile(args.inputDataCollection)

if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(args.outputDir)
    

NENT = 1
features_val = [fval[::NENT] for fval in traind.getAllFeatures()]
labels_val=traind.getAllLabels()[0][::NENT,:]
#weights_val=traind.getAllWeights()[0][::NENT]
#spectators_val = traind.getAllSpectators()[0][::NENT,0,:]
print features_val[0].shape
print labels_val.shape

#X_train_val = features_val[0][:,0,:]
#y_train_val = labels_val[:,1]
#print X_train_val.shape
#print y_train_val.shape

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(features_val[0], labels_val, test_size=0.2, random_state=42)
print X_train_val.shape
print y_train_val.shape
print X_test.shape
print y_test.shape
X_train_val = [X_train_val]
X_test = [X_test]


from DeepJet_models_ResNet import deep_model_doubleb
#from models.dense import dense_model

from keras.optimizers import Adam, Nadam
from keras.layers import Input

#keras_model = dense_model([Input(shape=(1,27,))], 2, 0, dropoutRate=0.1)
keras_model = deep_model_doubleb([Input(shape=(1,27,))], 2, 0)

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

keras_model.fit(X_train_val, y_train_val, batch_size = 1024, epochs = 1000,
                validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)
