import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')
from keras.models import load_model, Model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from root_numpy import array2root
import pandas as pd
seed = 42
np.random.seed(seed)


# In[4]:

def makeRoc(testd, model, outputDir):

    print 'in makeRoc()'
    
    # let's use only first 10000000 entries
    NENT = 10000000
    features_val = [fval[:NENT] for fval in testd.getAllFeatures()]
    labels_val=testd.getAllLabels()[0][:NENT,:]
    #weights_val=testd.getAllWeights()[0][:NENT]
    spectators_val = testd.getAllSpectators()[0][:NENT,0,:]
    
    X_train_val, X_test, y_train_val, y_test, z_train_val, z_test = train_test_split(features_val[0], labels_val, spectators_val,
                                                                                     test_size=0.2, random_state=42)
    for sample in ['train_val','test']:
        
        features_val = [ eval('X_%s'%sample) ]
        labels_val = eval('y_%s'%sample)
        spectators_val = eval('z_%s'%sample)

        df = pd.DataFrame(spectators_val)
        df.columns = ['fj_pt',
                  'fj_eta',
                  'fj_sdmass',
                  'fj_n_sdsubjets',
                  'fj_doubleb',
                  'fj_tau21',
                  'fj_tau32',
                  'npv',
                  'npfcands',
                  'ntracks',
                  'nsv']
        
        predict_test = model.predict(features_val)
        df['fj_isH'] = labels_val[:,1]
        df['fj_deepdoubleb'] = predict_test[:,1]
        df = df[(df.fj_sdmass > 40) & (df.fj_sdmass < 200) & (df.fj_pt > 300) &  (df.fj_pt < 2500)]
    
        print(df.iloc[:10])

        fpr, tpr, threshold = roc_curve(df['fj_isH'],df['fj_deepdoubleb'])
        dfpr, dtpr, threshold1 = roc_curve(df['fj_isH'],df['fj_doubleb'])

        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]

        value = 0.01 # 1% mistag rate
        idx, val = find_nearest(fpr, value)
        deepdoublebcut = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcut,100*val))
    
        auc1 = auc(fpr, tpr)
        auc2 = auc(dfpr, dtpr)
        
        plt.figure()       
        plt.plot(tpr,fpr,label='deep double-b, auc = %.1f%%'%(auc1*100))
        plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(auc2*100))
        plt.semilogy()
        plt.xlabel("H(bb) efficiency")
        plt.ylabel("QCD mistag rate")
        plt.ylim(0.001,1)
        plt.grid(True)
        plt.legend()
        plt.savefig(outputDir+"ROC_%s.pdf"%sample)
        
        for col in df.columns:
            plt.figure()
            plt.hist(df[col], bins = 100, weights = 1-df['fj_isH'], alpha=0.5,label='QCD',normed=True)
            plt.hist(df[col], bins = 100, weights = df['fj_isH'], alpha=0.5,label='H(bb)',normed=True)
            plt.xlabel(col)
            plt.legend(loc='upper right')
            plt.savefig(outputDir+'/'+col+'_'+sample+'.pdf')
    
    return df

os.environ['CUDA_VISIBLE_DEVICES'] = ''

inputDir = 'train_deep_simple_all/'
inputModel = '%s/KERAS_check_best_model.h5'%inputDir
outputDir = inputDir.replace('train','out')

inputDataCollection = '/data/shared/BumbleB/convert_deepDoubleB_simple_all/dataCollection.dc'

if os.path.isdir(outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(outputDir)

model=load_model(inputModel, custom_objects=global_loss_list)
    

#intermediate_output = intermediate_layer_model.predict(data)

#print(model.summary())
    
from DataCollection import DataCollection
    
testd=DataCollection()
testd.readFromFile(inputDataCollection)
    
df = makeRoc(testd, model, outputDir)


# In[ ]:

# let's use only first 10000000 entries
NENT = 10000000
features_val = [fval[:NENT] for fval in testd.getAllFeatures()]


# In[ ]:

print model.summary()

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


import json
inputLogs = '%s/full_info.log'%inputDir
f = open(inputLogs)
myListOfDicts = json.load(f, object_hook=_byteify)
myDictOfLists = {}
for key, val in myListOfDicts[0].iteritems():
    myDictOfLists[key] = []
for i, myDict in enumerate(myListOfDicts):
    for key, val in myDict.iteritems():
        myDictOfLists[key].append(myDict[key])
val_loss = np.asarray(myDictOfLists['val_loss'])
loss = np.asarray(myDictOfLists['loss'])
plt.figure()
plt.plot(val_loss, label='validation')
plt.plot(loss, label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("%s/loss.pdf"%outputDir)



