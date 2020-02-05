from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import activations
from tensorflow.keras import initializers




def CNN(N_output=10, kernel_size=(3,3), stride=1, pool_size=(2,2), data_shape=(32,32,1), learning_rate=1E-4, decay_rate=1E-4): 
    act = activations.relu
    padding = 'same'

    input_img= Input(shape=data_shape)
    layer_1 = Conv2D(filters=64, kernel_size=(5,5), activation=act, strides=stride, padding=padding)(input_img)
    layer_1 = MaxPooling2D(pool_size=(2,2), padding=padding)(layer_1)

    layer_2 = Conv2D(filters=32, kernel_size=(9,9), activation=act, strides=stride, padding=padding)(layer_1)
    layer_2 = MaxPooling2D(pool_size=(4,4), padding=padding)(layer_2)

    layer_3 = Conv2D(filters=8, kernel_size=(3,3), activation=act, strides=stride, padding=padding)(layer_2)
    layer_3 = MaxPooling2D(pool_size=(2,2), padding=padding)(layer_3)

    layer_f = Flatten()(layer_3)
    layer_d = Dense(units=N_output, activation=act)(layer_f)

    model = Model(input_img, layer_d)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

    model.compile(loss=losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
    return model






def attentionCNN(N_output=10, attention=True, kernel_size=(3,3), stride=1, pool_size=(2,2), data_shape=(32,32,1), learning_rate=1E-4, decay_rate=1E-4):
    act = activations.relu
    padding = 'same'

    input_img= Input(shape=data_shape)
    layer_1 = Conv2D(filters=64, kernel_size=kernel_size, activation=act, strides=stride, padding=padding)(input_img)
    layer_1 = MaxPooling2D(pool_size=pool_size, padding=padding)(layer_1)

    layer_2 = Conv2D(filters=32, kernel_size=kernel_size, activation=act, strides=stride, padding=padding)(layer_1)
    layer_2 = MaxPooling2D(pool_size=pool_size, padding=padding)(layer_2)

    layer_3 = Conv2D(filters=8, kernel_size=kernel_size, activation=act, strides=stride, padding=padding)(layer_2)
    layer_3 = MaxPooling2D(pool_size=pool_size, padding=padding)(layer_3)
    
    layer_f = Flatten()(layer_3)
    layer_d = Dense(units=N_output, activation=act)(layer_f)

    if attention:
        # https://github.com/thushv89/attention_keras
        from utility.attention import AttentionLayer

        attn_layer = AttentionLayer(name='attention_layer')
        attn_out,_ = attn_layer([layer_3, layer_d])

        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([layer_d, attn_out])

        dense = Dense(fr_vsize, activation='relu', name='last_layer')
        dense_time = TimeDistributed(dense, name='time_distributed_layer')
        decoder_pred = dense_time(decoder_concat_input)

        model = Model(input_img, decoder_pred)
    else:
        model = Model(input_img, layer_d)



    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

    model.compile(loss=losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
    return model


# encoder_outputs - Sequence of encoder ouptputs returned by the RNN/LSTM/GRU (i.e. with return_sequences=True)
# decoder_outputs - The above for the decoder
# attn_out        - Output context vector sequence for the decoder. This is to be concat with the output of decoder (refer model/nmt.py for more details)
# attn_states     - Energy values if you like to generate the heat map of attention (refer model.train_nmt.py for usage)








