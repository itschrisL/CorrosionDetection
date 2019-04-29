import tensorflow as tf
#import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import functools

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd


class CorrosionDetectionModel:

    def __init__(self):
        # super(CorrosionDetectionModel, self).__init__()

        self.input_shape = (256, 256, 3)
        self.steps_per_epoch = 5
        self.epochs = 5
        self.batch_size = 1

        self.saved_weights_path = "./saved_weights.hdf5"
        self.cp = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=self.saved_weights_path,
                                            monitor='val_dice_loss',
                                            save_best_only=True,
                                            verbose=1)
        # Create model
        self.model = self.init_model()

    # Define the model first
    def init_model(self):
        inputs, outputs = self.create_layers()

        model = models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(optimizer='adam',
                      loss=self.bce_dice_loss,
                      metrics=[self.dice_loss])

        model.summary()
        return model

    def create_layers(self):
        inputs = layers.Input(shape=self.input_shape)  # 256
        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)  # 128
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)  # 64
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)  # 32
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)  # 16
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)  # 8
        center = self.conv_block(encoder4_pool, 1024)  # center
        decoder4 = self.decoder_block(center, encoder4, 512)  # 16
        decoder3 = self.decoder_block(decoder4, encoder3, 256)  # 32
        decoder2 = self.decoder_block(decoder3, encoder2, 128)  # 64
        decoder1 = self.decoder_block(decoder2, encoder1, 64)  # 128
        decoder0 = self.decoder_block(decoder1, encoder0, 32)  # 256
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
        return inputs, outputs

    # Train the model by giving the training an labels
    def train_model(self, training_set, training_labels):
        print("Number of training data: " + str(len(training_set)))

        if len(training_set) != len(training_labels):
            print("ERROR: number of training data and labels need to be the same length")
            return

        history = self.model.fit(training_set,
                                 training_labels,
                                 steps_per_epoch=1,
                                 epochs=self.epochs)
        self.show_history(history)

    # Helper method to display the results of training
    def show_history(self, history):
        dice = history.history['dice_loss']
        val_dice = history.history['val_dice_loss']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, dice, label='Training Dice Loss')
        plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Dice Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    # ====== Helper methods to calculate the dice score =====
    # Get dice coeff
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    # get dice loss
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    # get bce dice loss
    def bce_dice_loss(self, y_true, y_pred):
        loss = losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss
