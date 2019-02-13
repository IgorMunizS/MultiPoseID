import keras

from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout, Activation

def PRN(height, width, node_count):
    input = Input(shape=(height, width, 18))
    y = Flatten()(input)
    x = Dense(node_count, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 18, activation='relu')(x)
    x = keras.layers.Add()([x, y])
    x = keras.layers.Activation('softmax')(x)
    x = Reshape((height, width, 18))(x)
    model = Model(inputs=input, outputs=x)
    # print(model.summary())
    return model


def PRN_Seperate(height, width, node_count):
    input = Input(shape=(height, width, 18))
    y = Flatten(name="prn_flatten")(input)
    x = Dense(node_count, activation='relu', name="prn_dense_1")(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 18, activation='relu', name="prn_dense_2")(x)
    x = keras.layers.Add(name="prn_dense1_add_dense2")([x, y])
    out = []
    start = 0
    end = width * height

    for i in range(18):
        o = keras.layers.Lambda(lambda x: x[:, start:end], name="prn_lambda_" + str(i))(x)
        o = Activation('softmax', name="prn_activation_" + str(i))(o)
        out.append(o)
        start = end
        end = start + width * height

    x = keras.layers.Concatenate(name="prn_concat")(out)
    x = Reshape((height, width, 18), name="prn_reshape")(x)
    model = Model(inputs=input, outputs=x)
    # print(model.summary())
    return model