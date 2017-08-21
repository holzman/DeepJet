

from training_base import training_base
from Losses import loss_NLL

#also does all the parsing
train=training_base(testrun=False)


if not train.modelSet():
    from DeepJet_models_ResNet import resnet_model_doubleb_sv
    
    train.setModel(resnet_model_doubleb_sv)
    
    train.compileModel(learningrate=0.0004,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])
    


model,history,callbacks = train.trainModel(nepochs=1000, 
                                 batchsize=4096, 
                                 stop_patience=10, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)
