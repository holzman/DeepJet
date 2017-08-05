
import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training_base_compare import makeAllRocs
from training_base_compare import training_base_compare
#from Losses import loss_NLL
from models import dense_model
from models import dense_model_nLayers

# input and output directories
rocOutDir = 'RocTest/'

singleRun = True
singleInputDir = '../convertFromRoot/deepDoubleB_ALL_SV/dataCollection.dc'
singleRunOutDir = 'singleTest/'

testLearn = False
learnInputDir = '../convertFromRoot/deepDoubleB_ALL_Base/dataCollection.dc'
learnOutputDir = 'deepDoubleBcompareLearningRate'
learningrates = [.001,.002,.003,.004,.005,.006,.007,.008,.009,.010]

testLayers = False
layersInputDir = '../convertFromRoot/deepDoubleB_ALL_Base/dataCollection.dc'
layersOutputDir = 'deepDoubleBcompareLayers'
layersList = range(2,10)

testVariables = False
variablesInputDir = '../convertFromRoot/deepDoubleB_ALL_'
variablesOutputDir = 'deepDoubleBcompareVariables'
variablesList = ['Base','SV']

if(singleRun):
	training1 = training_base_compare(testrun = True,inputDataCollection = singleInputDir,outputDir = singleRunOutDir)

	training1.setModel(dense_model,dropoutRate=0.1)

	training1.compileModel(learningrate=0.003,
			   loss=['categorical_crossentropy'],
			   metrics=['accuracy'])
			   
	model,history,callbacks = training1.trainModel(nepochs=1000, 
							 batchsize=250, 
							 stop_patience=100, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)
							 
	training1.loadModel(training1.outputDir + "KERAS_check_best_model.h5")

	training1.makeRoc(callbacks)

if(testLearn):
	trainingList = []
	names = []
	learnFileOut = "testLearningRateRocs.pdf"
	for i in range(len(learningrates)):
		learning = learningrates[i]
		outputDirectory = learnOutputDir + str(learning)

		training = training_base_compare(testrun=True, inputDataCollection = learnInputDir, outputDir = outputDirectory)
		training.setModel(dense_model,dropoutRate=0.1)
	
		training.compileModel(learningrate=learning,
			   				loss=['categorical_crossentropy'],
			   				metrics=['accuracy'])
			   				
		model,history,callbacks = training.trainModel(nepochs=100, 
							 batchsize=250, 
							 stop_patience=20, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)
							 
		training.loadModel(training.outputDir + "KERAS_check_best_model.h5")
	
		trainingList.append(training)
		names.append('learningRate = %.3f' %(learning))
		training.makeRoc(callbacks)

	makeAllRocs(trainingList,names,rocOutDir,learnFileOut)

	
if(testLayers):
	trainingList = []
	names = []
	layersFileOut = "testLayersRocs.pdf"
	for i in layersList:
		layers = i
		outputDirectory = layersOutputDir + str(layers)

		training = training_base_compare(testrun=True, inputDataCollection = layersInputDir, outputDir = outputDirectory)
		training.setModel(dense_model_nLayers,nLayers=layers,dropoutRate=0.1)
	
		training.compileModel(learningrate=0.002,
			   				loss=['categorical_crossentropy'],
			   				metrics=['accuracy'])
			   				
		model,history,callbacks = training.trainModel(nepochs=100, 
							 batchsize=250, 
							 stop_patience=15, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)
							 
		training.loadModel(training.outputDir + "KERAS_check_best_model.h5")
	
		trainingList.append(training)
		names.append('nLayers = %.3f' %(layers))
		training.makeRoc(callbacks)

	
	makeAllRocs(trainingList,names,rocOutDir,layersFileOut)

if(testVariables):
	trainingList = []
	names = []
	variablesFileOut = "testVariablesRocs.pdf"
	for i in range(len(variablesList)):
		variables = variablesList[i]
		outputDirectory = variablesOutputDir + variables

		training = training_base_compare(testrun=True, inputDataCollection = variablesInputDir+variables+'/dataCollection.dc', outputDir = outputDirectory)
		training.setModel(dense_model,dropoutRate=0.1)
	
		training.compileModel(learningrate=0.003,
			   				loss=['categorical_crossentropy'],
			   				metrics=['accuracy'])
			   				
		model,history,callbacks = training.trainModel(nepochs=1000, 
							 batchsize=250, 
							 stop_patience=100, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)
							 
		training.loadModel(training.outputDir + "KERAS_check_best_model.h5")
	
		trainingList.append(training)
		names.append('%s' %(variables))
		training.makeRoc(callbacks)

	makeAllRocs(trainingList,names,rocOutDir,variablesFileOut)





    
