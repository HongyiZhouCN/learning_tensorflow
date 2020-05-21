import tensorflow as tf
from data_loader import ImgLoader
from model import MultiTaskCNN


Path_img = "F:/learning_tensorflow/data/formatted_trainval/shanghaitech_part_A_patches_9/train"
Path_den = "F:/learning_tensorflow/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"
train_img = ImgLoader(Path_img, Path_den, 32)
model = MultiTaskCNN()
model.compile(optimizer='adam', loss='mse')
history = model.fit(x=train_img.data_generator_2(), epochs =1, steps_per_epoch = 10, verbose=1)