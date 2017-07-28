


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
try:
    imp.find_module('setGPU')
    print('running on GPU')
    import setGPU
except:
    found = False
    
# some private extra plots
#from  NBatchLogger import NBatchLogger

import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import keras
#zero padding done before
#from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras.optimizers import SGD

## to call it from cammand lines
import sys
import os
from argparse import ArgumentParser
import shutil

# argument parsing and bookkeeping
from Losses import *

class training_base_compare(object):
    
    def __init__(self, 
                 splittrainandtest=0.8,
                 useweights=False,
                 testrun=False,
        		 inputDataCollection = "",
        		 outputDir = ""):
        
		self.keras_inputs=[]
		self.keras_inputsshapes=[]
		self.keras_model=None
		self.train_data=None
		self.val_data=None
		self.startlearningrate=None
		self.trainedepoches=0
		self.compiled=False
        
		self.inputData = os.path.abspath(inputDataCollection)
		self.outputDir=outputDir
        # create output dir
        
		isNewTraining=True
		if os.path.isdir(self.outputDir):
			shutil.rmtree(self.outputDir)
		os.mkdir(self.outputDir)
		self.outputDir = os.path.abspath(self.outputDir)
		self.outputDir+='/'
        
		from DataCollection import DataCollection
        #copy configuration to output dir
		if isNewTraining:
			djsource= os.environ['DEEPJET']
			shutil.copytree(djsource+'/modules/models', self.outputDir+'models')
			shutil.copyfile(sys.argv[0],self.outputDir+sys.argv[0])

		self.train_data=DataCollection()
		self.train_data.readFromFile(self.inputData)
		self.train_data.useweights=useweights
        
		if testrun:
			self.train_data.split(0.02)
            
		self.val_data=self.train_data.split(splittrainandtest)
        
		self.train_data.writeToFile(self.outputDir+'trainsamples.dc')
		self.val_data.writeToFile(self.outputDir+'valsamples.dc')


		shapes=self.train_data.getInputShapes()
        
		self.keras_inputs=[]
		self.keras_inputsshapes=[]
        
		for s in shapes:
			self.keras_inputs.append(keras.layers.Input(shape=s))
			self.keras_inputsshapes.append(s)
            
		if not isNewTraining:
			self.loadModel(self.outputDir+'KERAS_check_last_model.h5')
			self.trainedepoches=sum(1 for line in open(self.outputDir+'losses.log'))
        
    
        
    def setModel(self,model,**modelargs):
        if len(self.keras_inputs)<1:
            raise Exception('setup data first') #can't happen
        self.keras_model=model(self.keras_inputs,
                               self.train_data.getNClassificationTargets(),
                               self.train_data.getNRegressionTargets(),
                               **modelargs)
            
        
    def loadModel(self,filename):
        #import h5py
        #f = h5py.File(filename, 'r+')
        #del f['optimizer_weights']
        from keras.models import load_model
        self.keras_model=load_model(filename, custom_objects=global_loss_list)
        self.compiled=True
        
    def compileModel(self,
                     learningrate,
                     **compileargs):
        if not self.keras_model:
            raise Exception('set model first') #can't happen
        #if self.compiled:
        #    return
        from keras.optimizers import Adam
        self.startlearningrate=learningrate
        adam = Adam(lr=self.startlearningrate)
        self.keras_model.compile(optimizer=adam,**compileargs)
        self.compiled=True
        
    def saveModel(self,outfile):
		self.keras_model.save(self.outputDir+outfile)
		#import h5py
		#f = h5py.File(self.outputDir+outfile, 'r+')
		#del f['optimizer_weights']
		#f.close()
		return
        
    def trainModel(self, nepochs, batchsize,
                   stop_patience=300, 
                   lr_factor=0.5,
                   lr_patience=2, 
                   lr_epsilon=0.003, 
                   lr_cooldown=6, 
                   lr_minimum=0.000001,
                   maxqsize=20,
                   **trainargs):   
        
		#make sure tokens don't expire
		from tokenTools import checkTokens, renew_token_process
		from thread import start_new_thread
        
		checkTokens()
		start_new_thread(renew_token_process,())
        
		self.train_data.setBatchSize(batchsize)
		self.val_data.setBatchSize(batchsize)
        
		self.keras_model.save(self.outputDir+'KERAS_check_last_model.h5')
        
		from DeepJet_callbacks import DeepJet_callbacks
        
		callbacks=DeepJet_callbacks(stop_patience=stop_patience, 
                                    lr_factor=lr_factor,
                                    lr_patience=lr_patience, 
                                    lr_epsilon=lr_epsilon, 
                                    lr_cooldown=lr_cooldown, 
                                    lr_minimum=lr_minimum,
                                    outputDir=self.outputDir)
		nepochs=nepochs-self.trainedepoches
        
		self.keras_model.fit_generator(self.train_data.generator() ,
                            steps_per_epoch=self.train_data.getNBatchesPerEpoch(), 
                            epochs=nepochs,
                            callbacks=callbacks.callbacks,
                            validation_data=self.val_data.generator(),
                            validation_steps=self.val_data.getNBatchesPerEpoch(), #)#,
                            max_q_size=maxqsize,**trainargs)
        
        
		self.saveModel("KERAS_model.h5")
        
		return self.keras_model, callbacks.history, callbacks #added callbacks
        
        
    def makeRoc(self,callbacks):
		from sklearn.metrics import roc_curve, auc
		from root_numpy import array2root
		traind = self.train_data
		testd = self.val_data
		outputDir = self.outputDir
		model = self.keras_model
        
    	
    	# summarize history for loss for training and test sample
		plt.figure(1)
		plt.plot(callbacks.history.history['loss'])
		plt.plot(callbacks.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.outputDir+'learningcurve.pdf') 
		plt.close(1)

		plt.figure(2)
		plt.plot(callbacks.history.history['acc'])
		plt.plot(callbacks.history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('acc')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.outputDir+'accuracycurve.pdf')
		plt.close(2)
    	
		features_val=self.train_data.getAllFeatures()
		labels_val=self.train_data.getAllLabels()
		weights_val=self.train_data.getAllWeights()[0]
		print(self.train_data.getAllSpectators())
		spectator_val=self.train_data.getAllSpectators()[0][:,0,4]; print(spectator_val)
				
		predict_test = self.keras_model.predict(features_val)

		fpr, tpr, threshold = roc_curve(labels_val[0][:,1],predict_test[:,1])
		dfpr, dtpr, threshold1 = roc_curve(labels_val[0][:,1],spectator_val)
        
		auc1 = auc(fpr, tpr)
		auc2 = auc(dfpr, dtpr)
		
		plt.figure(3)       
		plt.plot(tpr,fpr,label='deep double-b, auc = %.1f%%'%(auc1*100))
		plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(auc2*100))
		plt.semilogy()
		plt.xlabel("H(bb) f")
		plt.ylabel("QCD mistag rate")
		plt.ylim(0.001,1)
		plt.grid(True); plt.legend();
		plt.savefig(self.outputDir+"test.pdf")
		plt.close(3); plt.figure(4); plt.hist(spectator_val, weights = labels_val[0][:,0]); plt.savefig(self.outputDir+"doubleb_qcd.pdf"); plt.close(4); plt.figure(5); plt.hist(spectator_val, weights = 1.-labels_val[0][:,0]); plt.savefig(self.outputDir+"doubleb_hbb.pdf"); plt.close(5)
    

# 		y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
# 		scores = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.2, 0.2, 1])
# 		fpr, tpr, thresholds = roc_curve(y, scores)
# 		print(fpr)
# 		print(tpr)
# 		print(thresholds)
#     	       
# 		plt.figure(1)       
# 		plt.plot(tpr,fpr,label='testing')
# 		plt.semilogy()
# 		plt.xlabel("b efficiency")
# 		plt.ylabel("BKG efficiency")
# 		plt.ylim(0.001,1)
# 		plt.grid(True)
# 		plt.savefig(self.outputDir+"test.pdf")
# 		plt.close(1)
		return
    	       
def makeAllRocs(trainingList,names,outputDir):
	from sklearn import metrics
	from root_numpy import array2root

	plt.figure(4)       
	plt.semilogy()
	plt.xlabel("signal efficiency")
	plt.ylabel("background efficiency")
	plt.ylim(0.001,1)
	plt.suptitle("Roc Testing")
	
	for i in range(len(trainingList)):
		training = trainingList[i]
		name = names[i]
		traind = training.train_data
		testd = training.val_data
		model = training.keras_model
		features_val=training.train_data.getAllFeatures()
		labels_val=training.train_data.getAllLabels()
		weights_val=training.train_data.getAllWeights()[0]
		
		predict_test = training.keras_model.predict(features_val)

		fpr, tpr, threshold = metrics.roc_curve(labels_val[0][:,0],predict_test[:,0])
		auc = metrics.auc(labels_val[0][:,0],predict_test[:,0],True)
		
		name = name + " auc = " +str(auc)
		plt.plot(tpr,fpr,label=name)
	
	plt.legend()

	plt.savefig(outputDir+"testAllRocs.pdf")
	plt.close(4)
	
	
	return
	
        
        
        
            
    
