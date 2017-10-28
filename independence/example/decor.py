# Simple example to present a methods to add a measure of independence w.r.t. a variable "a" to the loss.
import numpy as np
import sys
import setGPU
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from Losses import *
from keras import backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
from DataCollection import DataCollection
import matplotlib.pyplot as plt
plt.switch_backend('agg')

seed = 7
np.random.seed(seed)

# New idea for DNN loss, use moments (or other analytical functions) and histograms. Both easy from computational point of view
# this loss can be used in KERAS and uses the Keras backend "K", i.e. derivatives ect. already included
def loss_moment(y_in,x):
    
    # h is the histogram vector "one hot encoded" (5 bins in this case), techically part of the "truth" y
    h = y_in[:,0:5]
    y = y_in[:,5:]
    
    # The below counts the entries in the histogram vector
    h_entr = K.sum(h,axis=0)
    
    ## first moment ##
    
    # Multiply the histogram vectors with estimated probability x
    h_fill = h * x
    
    # Sum each histogram vector
    Sum =K.sum(h_fill,axis=0)
    
    # Divide sum by entries (i.e. mean, first moment)
    Sum = Sum/h_entr
    
    # Divide per vector mean by average mean
    Sum = Sum/K.mean(x)
    
    ## second moment
    
    x2 = x-K.mean(x)
    h_fill2 = h * x2*x2
    Sum2 =K.sum(h_fill2,axis=0)
    Sum2 = Sum2/h_entr
    Sum2 = Sum2/K.mean(x2*x2)
    
    ## the loss, sum RMS + two moments (or more). The RMS is downweighted.
    
    return  0.005*K.mean(K.square(y - x)) + K.mean(K.square(Sum-1)) + K.mean(K.square(Sum2-1))


# Modification to M. Stoye idea for DNN loss: use KL divergence between weighted and antiweighted histograms (D. Anderson, J. Duarte)
def loss_kldiv_numpy(y_in,x):
    
    # h is the histogram vector "one hot encoded" (5 bins in this case), techically part of the "truth" y
    h = y_in[:,0:5]
    y = y_in[:,5:]
    
    # The below counts the entries in the histogram vector
    h_entr = np.sum(h,axis=0)
    
    # Multiply the histogram vectors with estimated probability x
    h_fill = h*np.reshape(x, (10000,1))
    h_anti = h*np.reshape(1-x, (10000,1))
    
    return  0.005*np.mean(np.square(y - x)) + stats.entropy(h_fill, h_anti)

def main():
    
    inputDataCollection = '../../convertFromRoot/convert_20170717_ak8_deepDoubleB_init_train_val_fixQCD/dataCollection.dc'
    
    traind=DataCollection()
    traind.readFromFile(inputDataCollection)

    
    NENT = 1 # take all events
    features_val = [fval[::NENT] for fval in traind.getAllFeatures()]
    labels_val=traind.getAllLabels()[0][::NENT,:]
    spectators_val = traind.getAllSpectators()[0][::NENT,0,:]

    # OH will be the truth "y" input to the network
    # OH contains both, the actual truth per sample and the actual bin (one hot encoded) of the variable to be independent of
    OH = np.zeros((labels_val.shape[0],42))
    print labels_val.shape
    print labels_val.shape[0]
    
    for i in range(0,labels_val.shape[0]):
        # bin of a (want to be independent of a)
        OH[i,int((spectators_val[i,2]-40.)/4.)]=1
        # aimed truth (target) 
        OH[i,40] = labels_val[i,0]
        OH[i,41] = labels_val[i,1]

    # make a simple model:
    from DeepJet_models_ResNet import deep_model_doubleb_sv

    model = deep_model_doubleb_sv([Input(shape=(1,27,)),Input(shape=(5,14,))], 2, 0)
    model.compile(loss=loss_kldiv,optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    # batch size is huge because of need to evaluate independence


    from DeepJet_callbacks import DeepJet_callbacks
    
    callbacks=DeepJet_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir='train_all_b1024_newloss1/')
    model.fit(features_val, OH, batch_size=1024, epochs=200, 
              verbose=1, validation_split=0.2, shuffle = True, 
              callbacks = callbacks.callbacks)
    #model.fit(features_val, labels_val, batch_size=1024, epochs=200, 
    #          verbose=1, validation_split=0.2, shuffle = True, 
    #          callbacks = callbacks.callbacks)
    # get the truth:
    #output = model.predict(x)


if __name__ == "__main__":

    main()
    



