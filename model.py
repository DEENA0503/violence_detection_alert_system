import tensorflow as tf

def VD_model(tf, wgts='final_model12.weights.h5'):

    layers = tf.keras.layers
    models = tf.keras.models
    optimizers = tf.keras.optimizers
    num_classes = 2
    # loading the VGG19 model
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(160, 160, 3))

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())

    model = models.Sequential()
    model.add(layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)))
    # LSTM
    model.add(layers.LSTM(30, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))

    adam = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    model.load_weights(wgts)
    return model

