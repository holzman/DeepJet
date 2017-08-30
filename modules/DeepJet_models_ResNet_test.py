import keras

kernel_initializer = 'he_normal'
bn_axis = -1

def FC(data, num_hidden, act='relu', p=None, name=''):
    fc = keras.layers.Dense(num_hidden, activation=act, name='%s_relu' % name)(data)
    if not p:
        return fc
    else:
        dropout = keras.layers.Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout

def residual_unit(data, num_filter, stride, dim_match, name, height=1, bottle_neck=True, bn_mom=0.9):
    if bottle_neck:
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(1,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)

        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)

        bn3 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn3' % name)(conv2)
        act3 = keras.layers.Activation('relu', name='%s_relu3' % name)(bn3)
        conv3 = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv3' % name)(act3)
        if dim_match:
            shortcut = data
        else:
            shortcut = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_shortcut' % name)(act1)
        return keras.layers.add([conv3, shortcut])

    else:
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)
        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)
        if dim_match:
            shortcut = data
        else:
            shortcut = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_shortcut' % name)(act1)
        return keras.layers.add([conv2, shortcut])


def resnet(data, units, filter_list, num_classes, height=1, bottle_neck=True, bn_mom=0.9, name=''):
    num_stages = len(units)
    data = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, scale=False, name='%s_bn_data' % name)(data)   
    body = keras.layers.Conv1D(filters=filter_list[0], kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv0' % name)(data)

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], stride=(1 if i == 0 else 2,), dim_match=False,
                             height=height, name='%s_stage%d_unit%d' % (name, i + 1, 1), bottle_neck=bottle_neck)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], stride=(1,), dim_match=True, height=height,
                                 name='%s_stage%d_unit%d' % (name, i + 1, j + 2), bottle_neck=bottle_neck)

    bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=2e-5, name='%s_bn1' % name)(body)
    act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
    pool = keras.layers.GlobalAveragePooling1D(name='%s_pool' % name)(act1)
#     flat = keras.layers.Flatten(name='%s_flatten' % name)(pool)
    return pool

def get_subnet(num_classes, input_name, data, height=1, filter_list=[64, 128, 256, 512, 1024], bottle_neck=True, units=[3, 4, 6, 3]):
    return resnet(data,
                  units=units,
                  name=input_name,
                  filter_list=filter_list,
                  height=height,
                  num_classes=num_classes,
                  bottle_neck=bottle_neck)

def inception(data,nFilters =32,bn_mom=0.9,stride=1,name=''):
    
    bn = keras.layers.BatchNormalization(axis=bn_axis,input_shape = (data.shape[1],data.shape[2]), momentum=bn_mom,
                                            epsilon=2e-5,scale=False, name='%s_bn_data' % name)(data)
    act = keras.layers.Activation('relu', name='%s_relu1' % name)(bn)
    print act.shape

    conv1 = keras.layers.Conv1D(filters=nFilters, kernel_size=(1), strides=(stride), padding='same',
                            kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act)
    
    conv3 = keras.layers.Conv1D(filters=nFilters, kernel_size=(3), strides=(stride), padding='same',
                            kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv3' % name)(act)
    
    conv5 = keras.layers.Conv1D(filters=nFilters, kernel_size=(5), strides=(stride), padding='same',
                            kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv5' % name)(act)

    pool = keras.layers.MaxPooling1D(padding = 'same',name = '%s_pool' % name)(act)
    
    concat = keras.layers.concatenate([conv1,conv3,conv5,pool],name = '%s_concat' % name)
    
    return concat

def resnet_model_doubleb_sv_test(inputs, num_classes,num_regclasses, **kwargs):
    
    input_db = inputs[0]
    input_sv = inputs[1] 
            
    print input_db.shape
    print input_sv.shape

    #Create model here

    x_db = inception(input_db,nFilters = 32,stride = 1,name = 'db')
    
    x_sv = inception(input_sv,nFilters = 8,stride = 2,name = 'sv')
    x_sv = inception(x_sv,nFilters = 16,stride = 2,name = 'sv1')
    x_sv = inception(x_sv,nFilters = 32,stride = 2,name = 'sv2')
    
    concat = keras.layers.concatenate([x_db,x_sv],name = 'mix_concat0')
    print concat.shape
    
    x_all = inception(concat,nFilters=8,name = 'mix')
    x_all = inception(x_all,nFilters=16,name = 'mix1')
    x_all = inception(x_all,nFilters=32,name = 'mix2')
    
    pool = keras.layers.GlobalAveragePooling1D(name='final_pool')(x_all)
    #fc1 = FC(x_all, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(pool)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model
