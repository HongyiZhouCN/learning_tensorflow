import tensorflow as tf
import os
import numpy as np
import cv2

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 256])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


class ImgLoader:

    def __init__(self,Path_img,Path_den, batch_size):
        self.Path_img = Path_img
        self.Path_den = Path_den
        self.files_img = os.listdir(Path_img)
        self.files_den = os.listdir(Path_den)
        self.ImgList = []
        self.DenList = []
        self.bath_size = batch_size
        self.image_height = 192
        self.image_width = 256
        self.den_height = 192
        self.den_width = 256


    def data_generator_2(self):
        all_index = list(range(0, len(self.files_img)))
        start_index = 0
        while True:
            if start_index + self.bath_size >= len(all_index):
                np.random.shuffle(all_index)
                continue
            batch_input_image, batch_output_map = [],[]
            print('from',start_index,'to',start_index + self.bath_size)
            for index in range(start_index, start_index+ self.bath_size):
                input_img = load_and_preprocess_image(os.path.join(self.Path_img,self.files_img[all_index[index]]))
                output_den = np.genfromtxt(os.path.join(self.Path_den,self.files_den[all_index[index]]), delimiter=',')
                output_den.resize(192, 256)
                batch_input_image.append(input_img)
                batch_output_map.append(output_den)
            start_index += self.bath_size
            yield np.array(batch_input_image), np.array(batch_output_map)





