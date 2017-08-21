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
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def main(args):
 
    if os.path.isdir(args.outputDir):
        raise Exception('output directory must not exists yet')


    model=load_model(args.inputModel, custom_objects=global_loss_list)
    
    print model.summary()
    
    td=testDescriptor()
    
    from DataCollection import DataCollection
    
    testd=DataCollection()
    testd.readFromFile(args.inputDataCollection)
    
    makeRoc(testd, model, args.outputDir)

def makeRoc(testd, model, outputDir):
    from sklearn.metrics import roc_curve, auc
    from root_numpy import array2root
    import pandas as pd

    ## # summarize history for loss for training and test sample
    ## plt.figure(1)
    ## plt.plot(callbacks.history.history['loss'])
    ## plt.plot(callbacks.history.history['val_loss'])
    ## plt.title('model loss')
    ## plt.ylabel('loss')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'learningcurve.pdf') 
    ## plt.close(1)

    ## plt.figure(2)
    ## plt.plot(callbacks.history.history['acc'])
    ## plt.plot(callbacks.history.history['val_acc'])
    ## plt.title('model accuracy')
    ## plt.ylabel('acc')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'accuracycurve.pdf')
    ## plt.close(2)

    features_val=testd.getAllFeatures()
    labels_val=testd.getAllLabels()
    weights_val=testd.getAllWeights()[0]
    #print(testd.getAllSpectators())
    #spectator_val=testd.getAllSpectators()[0][:,0,4]
    df = pd.DataFrame(testd.getAllSpectators()[0][:,0,:])
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

    print(df.iloc[:10])

        
    predict_test = model.predict(features_val)
    df['fj_isH'] = labels_val[0][:,1]
    df['fj_deepdoubleb'] = predict_test[:,1]

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
    plt.savefig(outputDir+"test.pdf")
    
    plt.figure()
    bins = np.linspace(-1,1,70)
    plt.hist(df['fj_doubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_doubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("BDT double-b")
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"doubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,1,70)
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("deep double-b")
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"deepdoubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,2000,70)
    plt.hist(df['fj_pt'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_pt'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$p_{\mathrm{T}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"pt.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    plt.hist(df['fj_sdmass'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    df_passdoubleb = df[df.fj_doubleb > 0.9]
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_passdoubleb.pdf")
    
    plt.figure()
    bins = np.linspace(0,200,70)
    df_passdeepdoubleb = df[df.fj_deepdoubleb > deepdoublebcut]
    plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_passdeepdoubleb.pdf")
    
    return

if __name__ == "__main__":
    
    parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
    parser.add_argument('inputModel')
    parser.add_argument('inputDataCollection')
    parser.add_argument('outputDir')
    parser.add_argument('--labels', action='store_true', help='store true labels in the trees')
    parser.add_argument('--monkey_class', default='', help='allows to read the data with a different TrainData, it is actually quite dangerous if you do not know what you are doing')

    args = parser.parse_args()

    main(args)
    
