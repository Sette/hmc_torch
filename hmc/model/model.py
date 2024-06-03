import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, Concatenate, Input, Normalization

import numpy as np

class OutputNormalization(layers.Layer):
    def call(self, x, **kwargs):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def _load_weights(model, weights_path):
    print(model.summary())

    if weights_path and tf.io.gfile.exists(f"{weights_path}.index"):
        print(f"load weights for model {model.name}. weights_path={weights_path}")
        model.load_weights(weights_path)

    return model


def build_cnn(feature, input_shape):
    x: object = Normalization(input_shape=[input_shape, 1], axis=None)(feature)
    x = layers.Conv1D(128, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(32, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)

    return x


def build_classification(x, levels_size, dropout, input_shape=1024, name='default'):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/2), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/4), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(levels_size[name], activation='sigmoid', name=name+'_output')(x)

    return x


def build_model(levels_size: dict, sequence_size: int = 1280, dropout: float = 0.1) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size, 1)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024
    depth = len(levels_size)

    x: object = build_cnn(music, input_shape)

    # Construção das camadas sequencialmente
    prev_output = x
    outputs = [prev_output]
    for level in range(1, depth + 1):
        # Construir a camada atual
        current_input = prev_output if level == 1 else Concatenate(axis=1)([OutputNormalization()(prev_output), x])
        current_output = build_classification(current_input, levels_size, dropout,
                                              input_shape=fcn_size + levels_size[f'level{level}'], name=f'level{level}')

        # Atualizar a saída anterior para a próxima iteração
        prev_output = current_output
        outputs.append(prev_output)


    model = tf.keras.models.Model([music], [
        current_output,
    ], name="Essentia")

    #     _load_weights(model, weights_path)
    
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer='adam',
                   loss='binary_crossentropy', metrics=['accuracy'])

    return model
