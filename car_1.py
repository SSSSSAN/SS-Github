
#モジュールの読み込み
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import itertools

import keras.models
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import MaxPool2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam, RMSprop, SGD

#画像の読み込みとpathの設定
program_path        = Path(__file__).parent.resolve()   #.parent.resolve()：このプログラムのディレクトリ:src
parent_path         = program_path.parent.resolve()     #このプログラムのディレクトリの親ディレクトリ    :car
data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'

train_dir           = os.path.join(data_processed_path, 'train')
validation_dir      = os.path.join(data_processed_path, 'validation')
test_dir            = os.path.join(data_processed_path, 'test')

label               = os.listdir(test_dir)
display_dir         = data_path / 'display'

#カテゴリーを配列に格納
classes = ['Audi-a3','BMW-1-series','MINI-clubman','Mitsubishi-outlander','Nissan-370z']


#各種初期設定値
input_image_size    = 256
batch_size          = 32
n_categories        = len(label)
classes_num         = 5

steps_per_epoch     = 2
epochs              = 1
validation_steps    = 2

#モデルの設定
'''
def original_model():

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation="relu",input_shape=(input_image_size,input_image_size,3)))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(16,(3,3),activation="relu"))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(classes_num,activation="softmax"))
    
    model.summary()

    return model
'''

def VGG16_model():

    base_model = VGG16(
        include_top = False,
        weights = "imagenet",
        input_shape = (input_image_size, input_image_size, 3)
        )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_categories, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

#モデルの選択
model = VGG16_model()

#モデルの追加学習
model.trainable = True

set_trainable = False
for layer in model.layers:
    if layer.name == 'block5_conv1': #ここを変更して #block4_conv1など
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False 

model.summary()

#モデル設定後にcompile
model.compile(
    optimizer = Adam(lr=1e-5),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
    )

#モデルの保存
model.save('original_model.h5')

#画像の前処理
train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    height_shift_range=0.5,
    width_shift_range=0.5,
    channel_shift_range=5.0,
    brightness_range=[0.3,1.0],
    fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    height_shift_range=0.5,
    width_shift_range=0.5,
    channel_shift_range=5.0,
    brightness_range=[0.3,1.0],
    fill_mode='nearest'
    )
 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
    )

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
    )

#学習
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    )

#Confution Matrix
#https://datascience.stackexchange.com/questions/67424/confusion-matrix-results-in-cnn-keras
#https://qiita.com/kotai2003/items/e85f17d7213cf84e3bcd
def plot_confusion_matrix(cm, classes, normalize=False, cmap = plt.cm.Blues):
    plt.imshow(cm, cmap=cmap)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_VGG_1.png')

Y_pred = model.predict_generator(validation_generator, steps=len(validation_generator))
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validation_generator.classes, y_pred)
plot_confusion_matrix(cm, classes=classes)


#訓練時の正解率と損失値
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)

#プロット範囲設定(スムージング)
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

#正答率のプロット
plt_acc = plt.figure()
plt.plot(epochs, smooth_curve(acc),"bo",label="Training Acc")
plt.plot(epochs, smooth_curve(val_acc),"b",label="Validation Acc")
plt.legend()
plt.savefig('acc_VGG.png')
plt.close(plt_acc)

#損失値のプロット
plt_loss = plt.figure()
plt.plot(epochs,smooth_curve(loss),"bo",label="Training Loss")
plt.plot(epochs,smooth_curve(val_loss),"b",label="Validation Loss")
plt.legend()
plt.savefig('loss_VGG.png')
plt.close(plt_loss)

#モデルの評価
evaluate_value = model.evaluate_generator(test_generator)
print('test acc_original :', '{:.1%}'.format(evaluate_value[1]))

print('Learning by Original model finished!')

#predict model and display images
files     = os.listdir(display_dir)
n_display = min(49, len(files))
img       = random.sample(files,n_display)

plt.figure(figsize=(10,10))

for i in range(n_display):
    temp_img=load_img(os.path.join(display_dir,img[i]),target_size=(input_image_size,input_image_size))
    plt.subplot(5,7,i+1)
    plt.imshow(temp_img)

    #Images normalization
    temp_img_array=img_to_array(temp_img)
    temp_img_array=temp_img_array.astype('float32')/255.0
    temp_img_array=temp_img_array.reshape((1,input_image_size,input_image_size,3))

    #predict image
    img_pred=model.predict(temp_img_array)
    
    #print(str(round(max(img_pred[0]),2)))
    plt.title(label[np.argmax(img_pred)] + str(round(max(img_pred[0]),2)))
    
    #eliminate xticks,yticks
    plt.xticks([]),plt.yticks([])

print('All finished!')
plt.show()