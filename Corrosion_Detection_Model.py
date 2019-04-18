import tensorflow as tf


class CorrosionDetectionModel(tf.keras.Model):

    def __init__(self):
        super(CorrosionDetectionModel, self).__init__()
        self.model = self.init_model()

    def init_model(self):
        # TODO: Change this to our cnn
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model