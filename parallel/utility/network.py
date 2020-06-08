from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import activations
from tensorflow.keras import initializers





def CNN(N_output=10, filters=(64,32,8), data_shape=(32,32,1), learning_rate=1E-4, decay_rate=1E-4, 
        kernel_size=((3,3),(3,3),(3,3)), pool_size=((2,2),(2,2),(2,2)), pooling_type='max', dropout=0.0): 
    act = activations.relu
    padding = 'same'

    input_img= Input(shape=data_shape)
    layer_1 = Conv2D(filters=filters[0], kernel_size=kernel_size[0], activation=act, padding=padding)(input_img)
    if pooling_type=='avg':
        layer_1 = AveragePooling2D(pool_size=pool_size[0], padding=padding)(layer_1)
    else:
        layer_1 = MaxPooling2D(pool_size=pool_size[0], padding=padding)(layer_1)

    layer_2 = Conv2D(filters=filters[1], kernel_size=kernel_size[1], activation=act, padding=padding)(layer_1)
    if pooling_type=='avg':
        layer_2 = AveragePooling2D(pool_size=pool_size[1], padding=padding)(layer_2)
    else:
        layer_2 = MaxPooling2D(pool_size=pool_size[1], padding=padding)(layer_2)

    layer_3 = Conv2D(filters=filters[2], kernel_size=kernel_size[2], activation=act, padding=padding)(layer_2)
    if pooling_type=='avg':
        layer_3 = AveragePooling2D(pool_size=pool_size[2], padding=padding)(layer_3)
    else:
        layer_3 = MaxPooling2D(pool_size=pool_size[2], padding=padding)(layer_3)

    layer_f = Flatten()(layer_3)
    layer_f = Dropout(rate=dropout)(layer_f)
    layer_d = Dense(units=N_output, activation=act)(layer_f)

    model = Model(input_img, layer_d)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

    model.compile(loss=losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
    return model


