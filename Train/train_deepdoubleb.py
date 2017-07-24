
import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training_base import training_base
#from Losses import loss_NLL

skip = False

#if(not skip):
#also dows all the parsing
train=training_base(testrun=True)

from models import dense_model

train.setModel(dense_model,dropoutRate=0.1)

train.compileModel(learningrate=0.005,
			   loss=['categorical_crossentropy'],
			   metrics=['accuracy'])


model,history,callbacks = train.trainModel(nepochs=5, 
							 batchsize=250, 
							 stop_patience=300, 
							 lr_factor=0.5, 
							 lr_patience=10, 
							 lr_epsilon=0.0001, 
							 lr_cooldown=2, 
							 lr_minimum=0.0001, 
							 maxqsize=10)


## train.train_data = DataCollection



##Roc Testing

train.makeRoc(callbacks)



# 
# 
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




    
