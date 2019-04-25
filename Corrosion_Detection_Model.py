import tensorflow as tf



class CorrosionDetectionModel():

    def __init__(self):
        #super(CorrosionDetectionModel, self).__init__()
        self.model = self.init_model()
        self.input_shape = (256, 256, 3)
        self.epochs = 4
        self.batch_size = 1


    def init_model(self):
        # TODO: Change this to our cnn
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("WORKED")
        return model

    '''
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
        '''
