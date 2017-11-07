import keras
import numpy as np
import tensorflow as tf
from keras import backend as K


kernel_initializer = 'he_normal'
kernel_initializer_fc = 'lecun_uniform'



def FC(data, num_hidden, act='relu', p=None, name=''):
    if act=='leakyrelu':
        fc = keras.layers.Dense(num_hidden, activation='linear', name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data) # Add any layer, with the default of a linear squashing function                                                                                                                                                                                                           
        fc = keras.layers.advanced_activations.LeakyReLU(alpha=.001)(fc)   # add an advanced activation                                                                                                     
    else:
        fc = keras.layers.Dense(num_hidden, activation=act, name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data)
    if not p:
        return fc
    else:
        dropout = keras.layers.Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout


def crop(start, end):
    def slicer(x):
        return x[:,:,start:end]
        
    return keras.layers.Lambda(slicer)


def deep_model_removal_sv(inputs, num_classes,num_regclasses, **kwargs):


#ordering of sv variables
                         #0 'sv_ptrel',
                         #1 'sv_erel',
                         #2 'sv_phirel',
                         #3 'sv_etarel',
                         #4 'sv_deltaR',
                         #5 'sv_pt',
                         #6 'sv_mass',
                         #7 'sv_ntracks',
                         #8 'sv_normchi2',
                         #9 'sv_dxy',
                         #10 'sv_dxysig',
                         #11 'sv_d3d',
                         #12 'sv_d3dsig',
                         #13 'sv_costhetasvpv'
    

    removedVars = [0,1,5,6]

    input_db = inputs[0]
    input_sv = inputs[1]

    sv_shape = input_sv.shape

    passedVars = []
    start = 0
    end = 14
    index =0
    for i in removedVars:
        if i == start:
            start +=1
        if i > start:
            sliced = crop(start,i)(input_sv)
            print sliced.shape
            passedVars.append(sliced)
            start = i+1
    sliced = crop(start,end)(input_sv)
    passedVars.append(sliced)
    print passedVars
   
    cut_sv = keras.layers.concatenate(passedVars, axis = 2, name = 'cut_sv')
    
    x = keras.layers.Flatten()(input_db)

    sv = keras.layers.Flatten()(cut_sv)
    
    concat = keras.layers.concatenate([x, sv], name='concat')
    
    fc = FC(concat, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model
