import keras

kernel_initializer = 'he_normal'
kernel_initializer_fc = 'lecun_uniform'

bn_axis = -1
bn_momentum = 0.1
bn_eps = 0.001

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

def residual_unit(data, num_filter, stride, dim_match, name, height=1, bottle_neck=True, bn_mom=bn_momentum):
    if bottle_neck:
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(1,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)

        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=int(num_filter * 0.25), kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)

        bn3 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn3' % name)(conv2)
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
        bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn1' % name)(data)
        act1 = keras.layers.Activation('relu', name='%s_relu1' % name)(bn1)
        conv1 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1' % name)(act1)
        bn2 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn2' % name)(conv1)
        act2 = keras.layers.Activation('relu', name='%s_relu2' % name)(bn2)
        conv2 = keras.layers.Conv1D(filters=num_filter, kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2' % name)(act2)
        if dim_match:
            shortcut = data
        else:
            shortcut = keras.layers.Conv1D(filters=num_filter, kernel_size=(1,), strides=stride, padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_shortcut' % name)(act1)
        return keras.layers.add([conv2, shortcut])


def resnet(data, units, filter_list, num_classes, height=1, bottle_neck=True, bn_mom=bn_momentum, name=''):
    num_stages = len(units)
    data = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, scale=False, name='%s_bn_data' % name)(data)
    body = keras.layers.Conv1D(filters=filter_list[0], kernel_size=(3,), strides=(1,), padding='same',
                                    kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv0' % name)(data)

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], stride=(1 if i == 0 else 2,), dim_match=False,
                             height=height, name='%s_stage%d_unit%d' % (name, i + 1, 1), bottle_neck=bottle_neck)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], stride=(1,), dim_match=True, height=height,
                                 name='%s_stage%d_unit%d' % (name, i + 1, j + 2), bottle_neck=bottle_neck)

    bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_mom, epsilon=bn_eps, name='%s_bn1' % name)(body)
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

def resnet_model(inputs, num_classes,num_regclasses, **kwargs):

    input_jet = inputs[0]
    input_cpf = inputs[1]
    input_npf = inputs[2]
    input_sv = inputs[3]
    
    input_regDummy=inputs[4]
    
    reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    cpf = get_subnet(num_classes, data=input_cpf, input_name='Cpfcan', filter_list=[32, 64, 64, 128], bottle_neck=False, units=[2, 2, 2])
    npf = get_subnet(num_classes, data=input_npf, input_name='Npfcan', filter_list=[32, 32, 64, 64], bottle_neck=False, units=[2, 2, 2])
    sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[3, 3])

    concat = keras.layers.concatenate([input_jet, cpf, npf, sv], name='concat')
    fc1 = FC(concat, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(fc1)

    model = keras.models.Model(inputs=inputs, outputs=[output,reg])

    return model

def resnet_model_doubleb(inputs, num_classes,num_regclasses, **kwargs):

    print inputs[0].shape
    print inputs[0][:,0,:].shape
    input_db = inputs[0]
    input_pf = inputs[1]
    input_cpf = inputs[2]
    input_sv = inputs[3]

    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    
    #input_regDummy=inputs[5]
    
    #reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    
    pf = get_subnet(num_classes, data=input_pf, input_name='pfcand', filter_list=[32, 64, 64, 128], bottle_neck=False, units=[2, 2, 2])
    cpf = get_subnet(num_classes, data=input_cpf, input_name='cpfcand', filter_list=[32, 32, 64, 64], bottle_neck=False, units=[2, 2, 2])
    sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[3, 3])

    concat = keras.layers.concatenate([x, pf, cpf, sv], name='concat')
    fc1 = FC(concat, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(fc1)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model


def resnet_model_doubleb_sv_original(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]
    input_sv = inputs[1]

    print input_db.shape
    print input_sv.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
        
    #reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32, 64], bottle_neck=False, units=[3, 3])
    #sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32], bottle_neck=False, units=[1])
    print sv.shape
    
    concat = keras.layers.concatenate([x, sv], name='concat')
    
    fc1 = FC(concat, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(fc1)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

def resnet_model_doubleb_sv_simple(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]
    input_sv = inputs[1]

    print input_db.shape
    print input_sv.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
        
    #reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    sv = get_subnet(num_classes, data=input_sv, input_name='sv', filter_list=[32, 32], bottle_neck=False, units=[1])
    print sv.shape
    
    concat = keras.layers.concatenate([x, sv], name='concat')
    
    fc1 = FC(concat, 512, p=0.2, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(fc1)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

def deep_model_doubleb_sv(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]
    input_sv = inputs[1]

    x = keras.layers.Flatten()(input_db)

    sv = keras.layers.Flatten()(input_sv)
    
    concat = keras.layers.concatenate([x, sv], name='concat')
    
    fc = FC(concat, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    #fc = FC(fc, 100, p=0.1, name='fc4')
    #fc = FC(fc, 100, p=0.1, name='fc5')
    #fc = FC(fc, 100, p=0.1, name='fc6')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model


def deep_model_doubleb(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    
    fc = FC(x, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model


def deep_model_doubleb_batchnorm(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    #bn0 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_eps, name='input_bn0')(x)
    fc1 = keras.layers.Dense(16, activation='linear', name='fc1', kernel_initializer=kernel_initializer_fc)(x) #(bn1)
    bn1 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_eps, name='fc1_bn1')(fc1)
    act1 = keras.layers.Activation('relu', name='fc1_relu1')(bn1)                 
    dp1 = keras.layers.Dropout(rate=0.1, name='fc1_dropout')(act1)

    
    
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(dp1)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

def deep_model_doubleb_nobatchnorm(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    #bn0 = keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_eps, name='input_bn0')(x)
    fc = FC(x, 16, p=0.1, name='fc1')
    
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

   
def deep_model_full(inputs, num_classes,num_regclasses, **kwargs):

    print inputs[0].shape
    print inputs[0][:,0,:].shape
    input_db = inputs[0]
    input_pf = inputs[1]
    input_cpf = inputs[2]
    input_sv = inputs[3]

    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    
    #input_regDummy=inputs[5]
    
    #reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    
    pf = keras.layers.Flatten()(input_pf)
    cpf = keras.layers.Flatten()(input_cpf)
    sv = keras.layers.Flatten()(input_sv)

    concat = keras.layers.concatenate([x, pf, cpf, sv], name='concat')


    fc = FC(concat, 100, p=0.25, name='fc1')
    fc = FC(fc, 100, p=0.25, name='fc2')
    fc = FC(fc, 100, p=0.25, name='fc3')
    fc = FC(fc, 100, p=0.25, name='fc4')
    fc = FC(fc, 100, p=0.25, name='fc5')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)
                            
    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

   

def conv_model_full(inputs, num_classes,num_regclasses, **kwargs):

    print inputs[0].shape
    print inputs[0][:,0,:].shape
    input_db = inputs[0]
    input_pf = inputs[1]
    input_cpf = inputs[2]
    input_sv = inputs[3]

    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    
    #input_regDummy=inputs[5]
    
    #reg=keras.layers.Dense(2,kernel_initializer='ones',trainable=False,name='reg_off')(input_regDummy)

    pf = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                             kernel_initializer=kernel_initializer, use_bias=False, name='pf_conv1', 
                             activation = 'relu')(input_pf)
    pf = keras.layers.SpatialDropout1D(rate=0.1)(pf)
    pf = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
                             kernel_initializer=kernel_initializer, use_bias=False, name='pf_conv2', 
                             activation = 'relu')(pf)
    pf = keras.layers.SpatialDropout1D(rate=0.1)(pf)
    #pf = keras.layers.Flatten()(pf)
    pf = keras.layers.GRU(50,go_backwards=True,implementation=2)(pf)
    pf = keras.layers.Dropout(rate=0.1)(pf)

    cpf = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                             kernel_initializer=kernel_initializer, use_bias=False, name='cpf_conv1',
                             activation = 'relu')(input_cpf)
    cpf = keras.layers.SpatialDropout1D(rate=0.1)(cpf)
    cpf = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                             kernel_initializer=kernel_initializer, use_bias=False, name='cpf_conv2',
                             activation = 'relu')(cpf)
    cpf = keras.layers.SpatialDropout1D(rate=0.1)(cpf)
    #cpf = keras.layers.Flatten()(cpf)
    cpf = keras.layers.GRU(50,go_backwards=True,implementation=2)(cpf)
    cpf = keras.layers.Dropout(rate=0.1)(cpf)

    sv = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                             kernel_initializer=kernel_initializer, use_bias=False, name='sv_conv1',
                             activation = 'relu')(input_sv)
    sv = keras.layers.SpatialDropout1D(rate=0.1)(sv)
    sv = keras.layers.Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                             kernel_initializer=kernel_initializer, use_bias=False, name='sv_conv2',
                             activation = 'relu')(sv)
    sv = keras.layers.SpatialDropout1D(rate=0.1)(sv)
    #sv = keras.layers.Flatten()(sv)
    sv = keras.layers.GRU(50,go_backwards=True,implementation=2)(sv)
    sv = keras.layers.Dropout(rate=0.1)(sv)

    concat = keras.layers.concatenate([x, pf, cpf, sv], name='concat')


    fc = FC(concat, 100, p=0.1, name='fc1')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)
                            
    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

def deep_model_doubleb_1layer(inputs, num_classes,num_regclasses, **kwargs):

    
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    
    fc = FC(x, 64, p=0.1, name='fc1')
    fc = FC(fc, 32, p=0.1, name='fc2')
    fc = FC(fc, 32, p=0.1, name='fc3')
    output = keras.layers.Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model


def deep_model_doubleb_1layer(inputs, num_classes,num_regclasses, **kwargs):

    
    output = keras.layers.Dense(1, activation='linear', name='linear', kernel_initializer=kernel_initializer_fc)(inputs)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model



def deep_model_doubleb_2layer(inputs, num_classes,num_regclasses, **kwargs):

    fc = FC(inputs, 32, p=0.1, name='fc1')
    output = keras.layers.Dense(1, activation='linear', name='linear', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

