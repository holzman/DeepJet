
import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training_base_compare import makeAllRocs
from training_base_compare import training_base_compare
#from Losses import loss_NLL
from models import dense_model

#also dows all the parsing
#train=training_base(testrun=True)
#train2=training_base(testrun=True)

RocOutDir = "RocTest/"

#define trainings
training1 = training_base_compare(testrun=True,inputDataCollection='../convertFromRoot/convert_deepDoubleB/dataCollection.dc',outputDir ='test4')

#add all trainings to be compared
trainingList = [training1]
names = ['deepdoubleb']

# train each model
training1.setModel(dense_model,dropoutRate=0.1)

training1.compileModel(learningrate=0.003,
			   loss=['categorical_crossentropy'],
			   metrics=['accuracy'])

# visalize model
from keras.utils.vis_utils import plot_model
print training1.keras_model.summary()
plot_model(training1.keras_model,to_file='model.eps',show_shapes=True,show_layer_names=True)
			   
model,history,callbacks = training1.trainModel(nepochs=1000, 
							 batchsize=250, #128
							 stop_patience=100, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)



training1.loadModel(training1.outputDir + "KERAS_check_best_model.h5")

training1.makeRoc(callbacks)

#makeRocs
makeAllRocs(trainingList,names,RocOutDir)

# 
# Basic roc curve example
# y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
# scores = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.2, 0.2, 1])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores)
# 
# print(fpr)
# print(tpr)
# print(thresholds)
# 
# 
# plt.figure(1)
# 
# plt.plot(tpr,fpr,label='testing')
# 
# plt.semilogy()
# plt.xlabel("b efficiency")
# plt.ylabel("BKG efficiency")
# plt.ylim(0.001,1)
# plt.grid(True)
# plt.savefig("test.pdf")
# plt.close(1)

#roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)




    
