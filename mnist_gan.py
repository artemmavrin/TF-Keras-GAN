"""Generator and discriminator components for an MNIST GAN.

Architectures taken from https://www.tensorflow.org/tutorials/generative/dcgan
Permalink: https://github.com/tensorflow/docs/blob/dabcd8664c95e21b8c7f5b190d1ea9de87fcc369/site/en/tutorials/generative/dcgan.ipynb
"""

import tensorflow as tf


def make_generator(latent_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input([latent_dim]),
        tf.keras.layers.Dense(units=(7 * 7 * 256), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=5,
            padding='same',
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=5,
            strides=2,
            padding='same',
            use_bias=False,
            activation='sigmoid',
        ),
    ])


def make_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Input([28, 28, 1]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2,
                               padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2,
                               padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1),
    ])
