import tensorflow as tf
from tensorflow.keras.layers import Rescaling
# from tensorflow.keras.utils import plot_model
# from graphviz import Digraph

def VD_model(tf, wgts='final_model12.weights.h5'):

    # Rescaling(1./255, input_shape=(160, 160, 3)), # scale all the frames
    # Load the VGG19 model
    layers = tf.keras.layers
    models = tf.keras.models
    optimizers = tf.keras.optimizers
    num_classes = 2

    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(160, 160, 3))

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())

    model = models.Sequential()
    model.add(layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)))
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
    # Save and visualize the model
    # plot_model(model, to_file='violence_detection_model.png', show_shapes=True)
    # model.save("vd_model2.h5")
    

    # dot = Digraph()

    # # Add nodes
    # dot.node('Input', 'Input (Frames)')
    # dot.node('CNN', 'Spatial Feature Extraction\n(Time-distributed CNN)')
    # dot.node('LSTM1', 'LSTM(30)')
    # dot.node('LSTM2', 'LSTM(30)')
    # dot.node('Dense1', 'Dense(90)')
    # dot.node('Output', 'Output (1)')

    # # Add edges
    # dot.edge('Input', 'CNN')
    # dot.edge('CNN', 'LSTM1')
    # dot.edge('LSTM1', 'LSTM2')
    # dot.edge('LSTM2', 'Dense1')
    # dot.edge('Dense1', 'Output')

    # # Save and visualize
    # dot.render('model_diagram2', format='png')

    return model

