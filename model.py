import tensorflow as tf
import numpy as np

class MultiTaskCNN(tf.keras.Model):

    def __init__(self):     # you should define your layers in __init__
        super(MultiTaskCNN, self).__init__()
        self.shared_Conv2D_1 = tf.keras.layers.Conv2D(16, 9, padding = 'same', activation = 'relu')
        self.shared_Conv2D_2 = tf.keras.layers.Conv2D(32, 7, padding = 'same', activation = 'relu')
        self.density_Conv2D_1 = tf.keras.layers.Conv2D(20, 7, padding = 'same', activation = 'relu')
        self.density_MaxPooling_1 = tf.keras.layers.MaxPooling2D(2)
        self.density_Conv2D_2 = tf.keras.layers.Conv2D(40, 5, padding = 'same', activation = 'relu')
        self.density_MaxPooling_2 = tf.keras.layers.MaxPooling2D(2)
        self.density_Conv2D_3 = tf.keras.layers.Conv2D(20, 5, padding = 'same', activation='relu')
        self.density_Conv2D_4 = tf.keras.layers.Conv2D(10, 5, padding = 'same', activation='relu')
        self.output_Conv2D_1 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation ='relu')
        self.output_Conv2D_2 = tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu')
        self.output_DeConv2D_1 = tf.keras.layers.Conv2DTranspose(16, 4, strides = 2, padding = 'same')
        self.output_DeConv2D_2 = tf.keras.layers.Conv2DTranspose(1, 4, strides = 2, padding = 'same')


    def call(self, inputs, training=False):  # you should implement the model's forward pass in call
        x = self.shared_Conv2D_1(inputs)
        x = self.shared_Conv2D_2(x)
        x = self.density_Conv2D_1(x)
        x = self.density_MaxPooling_1(x)
        x = self.density_Conv2D_2(x)
        x = self.density_MaxPooling_2(x)
        x = self.density_Conv2D_3(x)
        x = self.density_Conv2D_4(x)
        x = self.output_Conv2D_1(x)
        x = self.output_Conv2D_2(x)
        x = self.output_DeConv2D_1(x)
        x = self.output_DeConv2D_2(x)
        print(x.shape)
        return x


my_model = MultiTaskCNN()
my_model.build(input_shape = (32,192,256,3))
test_input = np.random.rand(32, 192, 256, 3)
my_model.call(test_input)


