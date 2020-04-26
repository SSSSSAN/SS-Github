import os
import random
from pathlib import Path
import tkinter
import tkinter.filedialog
from tkinter import ttk,N,E,S,W,font
import matplotlib.pyplot as plt
import numpy as np

import keras.models
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import MaxPool2D #MAxPool2D equal MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint


program_path        = Path(__file__).parent.resolve()       #.parent.resolve()：このプログラムのディレクトリ:src
parent_path         = program_path.parent.resolve()         #このプログラムのディレクトリの親ディレクトリ    :deeplearning_for_photos
data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'

train_dir           = os.path.join(data_processed_path, 'train')
validation_dir      = os.path.join(data_processed_path, 'validation')

train_cats_dir      = os.path.join(train_dir, 'cats')       #train用の猫画像のディレクトリ
train_dogs_dir      = os.path.join(train_dir, 'dogs')       #train用の犬画像のディレクトリ
validation_cats_dir = os.path.join(validation_dir, 'cats')  #validation用の猫画像のディレクトリ
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  #validation用の犬画像のディレクトリ

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'


#モデルの設定
def create_model():

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(512,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    
    model.summary()

    return model

#モデル設定後にcompile
model = create_model()
model.compile(
    optimizer = RMSprop(lr=1e-4),
    loss = 'binary_crossentropy',
    metrics = ["accuracy"]
)

#画像の前処理
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)

#学習
history = model.fit_generator(train_generator,
                            steps_per_epoch=50,
                            epochs=30,
                            validation_data=validation_generator,
                            validation_steps=50)

#訓練時の損失値と正解率をプロット
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)

#正答率
plt_acc = plt.figure()
plt.plot(epochs, acc,"bo",label="Training Acc")
plt.plot(epochs, val_acc,"b",label="Validation Acc")
plt.legend()
plt.savefig('acc.png')
plt.close(plt_acc)

#損失値
plt_loss = plt.figure()
plt.plot(epochs,loss,"bo",label="Training Loss")
plt.plot(epochs,val_loss,"b",label="Validation Loss")
plt.legend()
plt.savefig('loss.png')
plt.close(plt_loss)