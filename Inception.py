from keras.optimizers import SGD
from sklearn.utils import shuffle
import numpy as np
import os
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
from keras.applications.inception_v3 import InceptionV3,GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM,Dense
from keras.models import  Model

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
num_frames_per_clip=2
batch_size=16


def bring_data_from_directory():
  datagen = ImageDataGenerator(rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
          'data/train',
          target_size=(299, 299),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['violent','norm'])

  validation_generator = datagen.flow_from_directory(
          'data/test',
          target_size=(299, 299),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['violent','norm'])

  return train_generator,validation_generator



#加载VGG模型
def load_InceptionV3_model():
  base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

  print("Model loaded..!")
  #print(base_model.summary())
  return base_model



if __name__ == '__main__':
    #加载模型

    base_model=load_InceptionV3_model()
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    #获取训练数据
    print('load data')
    train,validation=bring_data_from_directory()
    #获取测试数据

    print('start training')

    sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0)]
    hist=model.fit_generator(train, epochs=5, validation_steps=10,validation_data=validation,steps_per_epoch=500)
    print(hist)
    #存储模型的权重
    model.save_weights('incepweights/my_inception.h5')
