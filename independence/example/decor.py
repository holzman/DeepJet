# Simple example to present a methods to add a measure of independence w.r.t. a variable "a" to the loss.
import numpy as np
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except:
    pass
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping

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
def loss_kldiv(y_in,x):
    
    # h is the histogram vector "one hot encoded" (20 bins in this case), techically part of the "truth" y
    h = y_in[:,0:20]
    y = y_in[:,20:]
    
    # The below counts the entries in the histogram vector
    h_entr = K.sum(h,axis=0)
    
    ## first moment ##
    
    
    # Multiply the histogram vectors with estimated probability x, and 1-x
    h_fill_anti = K.dot(K.transpose(h), x) # * (1-y)   
    h_fill = h_fill_anti[:,0]    
    h_fill = h_fill / K.sum(h_fill,axis=0)
    h_anti = h_fill_anti[:,1]
    h_anti = h_anti / K.sum(h_anti,axis=0)
    
    return categorical_crossentropy(y, x) + 0.005*kullback_leibler_divergence(h_fill, h_anti)

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
    
    inputDataCollection = '/cms-sc17/convert_20170717_ak8_deepDoubleB_simple_train_val/dataCollection.dc'
    
    traind=DataCollection()
    traind.readFromFile(inputDataCollection)

    
    NENT = 10000 # take first 10k
    features_val = [fval[:NENT] for fval in traind.getAllFeatures()]
    labels_val=traind.getAllLabels()[0][:NENT,:]
    spectators_val = traind.getAllSpectators()[0][:NENT,0,:]

    # OH will be the truth "y" input to the network
    # OH contains both, the actual truth per sample and the actual bin (one hot encoded) of the variable to be independent of
    OH = np.zeros((NENT,22))
    for i in range(0,NENT):
        # bin of a (want to be independent of a)
        OH[i,int((spectators_val[i,2]-40.)/8.)]=1
        # aimed truth (target) 
        OH[i,20] = labels_val[i,0]
        OH[i,21] = labels_val[i,1]

    x = features_val
    ## WARNING
    # y = np.vstack((c,a)).T
    #print (y.shape, ' ',x.shape)
    
    h = OH[:,0:20]
    y = OH[:,20:]
    h_entr = np.sum(h,axis=0)
    print ('sum', h_entr)

    # make a simple model:

    model = dense_model([Input(shape=(1,27,))])
    model.compile(loss=loss_kldiv,optimizer='adam', metrics=['accuracy'])

    # batch size is huge because of need to evaluate independence
    model.fit(x, OH, batch_size=NENT, nb_epoch=100, verbose=1, validation_split=0.2)
    # get the truth:
    output = model.predict(x)


def dense_model(Inputs,dropoutRate=0.1):
    """                      
    Dense matrix, defaults similat to 2016 training                                                                           
    """
    x = Flatten()(Inputs[0])
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    predictions = Dense(2, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


if __name__ == "__main__":

    main()
    



