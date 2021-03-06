#© 2019 Horie Yuki
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

#print(device_lib.list_local_devices())

# coding: UTF-8
import os
import random
from pathlib import Path

import tkinter
import tkinter.filedialog
from tkinter import ttk,N,E,S,W,font

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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

program_path = Path(__file__).parent.resolve() #.parent.resolve()：このプログラムのディレクトリ:src
parent_path = program_path.parent.resolve()    #このプログラムのディレクトリの親ディレクトリ    :deeplearning_for_photos

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'

result_path         = parent_path / 'results'

def chk_mkdir(paths):
    for path_name in paths:
        print(path_name)
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)
    return

def main():
    data_folder_path = t_folderpath.get()

    theme_dir        = Path(data_folder_path)
    #theme_dir        = os.path.basename(theme_dir)
    #theme_dir        =r'C:\Users\3ken\Desktop\data_science\pictures\actors'

    theme_name      =  os.path.basename(theme_dir)

    train_dir       = theme_dir / 'train'
    validation_dir  = theme_dir / 'validation'
    test_dir        = theme_dir / 'test'
    display_dir     = theme_dir / 'display'

    paths = [parent_path / 'models' / theme_name,
             parent_path / 'models' / theme_name / 'best_model',
             parent_path / 'models' / theme_name / 'checkpoint',
             parent_path / 'models' / theme_name / 'csvlogger',
                ]
    chk_mkdir(paths)

    label=os.listdir(test_dir)
    n_categories = len(label)

    print(label)
    print(n_categories)


    high_low = str(var_epoch.get())
    epoch_list = {'high':500, 'middle':50, 'low':10}
    n_epochs = epoch_list[high_low]

    net_type = var_model.get()
    #net_type = 'mobilenet'
    batch_size=8
    input_image_size = 224

    file_name = net_type +'_'  + theme_name +'_category_' + str(n_categories) + '_' + str(n_epochs) + 'eps_'

    def create_original_model():

        model = Sequential() # model = keras.models.Sequential()と等価

        model.add(Conv2D(32,3,input_shape=(input_image_size,input_image_size,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32,3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Conv2D(64,3))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(1.0))

        model.add(Dense(n_categories, activation='softmax'))

        return model

    def create_xception():
        base_model = Xception(
            include_top = False,
            weights = "imagenet",
            input_shape = None
        )

        #add new layers instead of FC networks

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        predictions = Dense(n_categories, activation = 'softmax')(x)

        # ネットワーク定義
        model = Model(inputs = base_model.input, outputs = predictions)
        print("{}層".format(len(model.layers)))


        #108層までfreeze
        for layer in model.layers[:108]:
            layer.trainable = False

            # Batch Normalization の freeze解除
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True

        #109層以降、学習させる
        for layer in model.layers[108:]:
            layer.trainable = True
        return model

    def create_Inception():
        base_model = InceptionV3(
            include_top = False,
            weights = "imagenet",
            input_shape = None
        )

        #add new layers instead of FC networks

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        predictions = Dense(n_categories, activation = 'softmax')(x)

        # ネットワーク定義
        model = Model(inputs = base_model.input, outputs = predictions)
        print("{}層".format(len(model.layers)))


        #108層までfreeze
        for layer in model.layers[:249]:
            layer.trainable = False

            # Batch Normalization の freeze解除
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True

        #109層以降、学習させる
        for layer in model.layers[249:]:
            layer.trainable = True
        return model

    def create_mobilenet():
        base_model = MobileNet(
        include_top = False,
        weights = "imagenet",
        input_shape = None
        )

        # 全結合層の新規構築
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        predictions = Dense(n_categories, activation = 'softmax')(x)

        # ネットワーク定義
        model = Model(inputs = base_model.input, outputs = predictions)
        print("{}層".format(len(model.layers)))

        # 72層までfreeze
        for layer in model.layers[:72]:
            layer.trainable = False

            # Batch Normalization の freeze解除
            if "bn" in layer.name:
                layer.trainable = True

        #73層以降、学習させる
        for layer in model.layers[72:]:
            layer.trainable = True

        return model


    def create_vgg16():
        base_model = VGG16(
        include_top = False,
        weights = "imagenet",
        input_shape = None
        )

        # 全結合層の新規構築
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        predictions = Dense(n_categories, activation = 'softmax')(x)

        # ネットワーク定義
        model = Model(inputs = base_model.input, outputs = predictions)
        print("{}層".format(len(model.layers)))

        # 17層までfreeze
        for layer in model.layers[:17]:
            layer.trainable = False

        # 18層以降を訓練可能な層として設定
        for layer in model.layers[17:]:
            layer.trainable = True

        return model

    print('choosed model is ', net_type)

    if net_type == 'xception':
        model = create_xception()
    elif net_type == 'Inception':
        model = create_Inception()
    elif net_type == 'mobilenet':
        model = create_mobilenet()
    elif net_type == 'original':
        model = create_original_model()
    elif net_type == 'vgg16':
        model = create_vgg16()
    else:
        print(net_type, ' のモデル名が間違っています。　VGG16を適用します')
        model = create_vgg16()

    # layer.trainableの設定後にcompile
    model.compile(
        optimizer = Adam(),
        loss = 'categorical_crossentropy',
        metrics = ["accuracy"]
    )

    model.summary()

    train_datagen=ImageDataGenerator(
        rescale=1.0/255,            #画素値を[0, 255]から[0, 1]に変更する
        shear_range=0.2,            #-0.2°~0.2°の間でランダムにせん断
        zoom_range=0.2,             #0.8~1.2倍の間でランダムに拡大縮小
        vertical_flip=True,         #ランダムに上下反転
        horizontal_flip=True,       #ランダムに左右反転
        height_shift_range=0.5,     #-0.5~0.5の間で上下並行移動
        width_shift_range=0.5,      #-0.5~0.5の間で左右並行移動
        channel_shift_range=5.0,    #-5~5の間でランダムに画素値を足す
        brightness_range=[0.3,1.0], #0.3~1.0の間でランダムに明度を足す
        fill_mode='nearest'         #処理した際の余白を埋める
        )

    validation_datagen=ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_image_size,input_image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(input_image_size,input_image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )


    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(parent_path / 'models' / theme_name /'checkpoint' / (net_type + '_model_{epoch:02d}_{val_loss:.2f}.h5') ),
        monitor='val_loss',
        period=150,
        verbose=1)

    print(net_type)
    history=model.fit_generator(train_generator,
                             epochs=n_epochs,
                             verbose=1,
                             validation_data=validation_generator,
                             callbacks=[model_checkpoint, CSVLogger(parent_path / 'models' / theme_name /'csvlogger' / (file_name+'.csv') ) ])


    #Confution Matrix and Classification Report
    #https://datascience.stackexchange.com/questions/67424/confusion-matrix-results-in-cnn-keras
    #https://qiita.com/kotai2003/items/e85f17d7213cf84e3bcd
    def plot_confusion_matrix(cm, classes, cmap):

        plt.imshow(cm, cmap=cmap)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'confuse_' + file_name + '.png'))
    
    #validation_generator.samples/batch_size, verbose=0
    Y_pred       = model.predict_generator(validation_generator)
    y_pred       = np.argmax(Y_pred, axis=1)
    true_class   = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    cm = confusion_matrix(true_class, y_pred)
    cmap = plt.cm.Blues
    plot_confusion_matrix(cm, classes=class_labels, cmap=cmap)


    #load model and weights
    #model = keras.models.load_model(parent_path / 'models' / theme_name   / file_name + '_' + str(round(test_score[1],2))+ '.h5')

    #model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])


    #data generate
    test_datagen=ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        height_shift_range=0.5,
        width_shift_range=0.5,
        channel_shift_range=5.0,
        brightness_range=[0.3,1.0],
        fill_mode='nearest'
        )

    test_generator=test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_image_size,input_image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )


    #evaluate model
    test_score=model.evaluate_generator(test_generator)
    print('\n test loss:',test_score[0])
    print('\n test_acc:',test_score[1])

    #save weights
    #model.save('Test.h5')
    #print(str(theme_dir /'model' / (file_name + '_' + str(round(test_score[1],2)) + '.h5')))
    if Booleanvar_savemodel.get() == True:
        os.chdir(parent_path / 'models' / theme_name)
        #try:
        model.save(str( (file_name + '_' + str(round(test_score[1],2)) + '.h5')))
        print('モデル保存完了')
        #except OSError:
        #print('OSError')
        #print('model.saveに失敗しました。')
        #else:
        #    print('model saveに失敗しました')

    #訓練時の正解率と損失値
    acc = history.history['acc']
    val_acc = history.history['val_acc']
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
    plt.plot(epochs, smooth_curve(acc),"go",label="Training Acc")
    plt.plot(epochs, smooth_curve(val_acc),"bo",label="Validation Acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid(color='gray', alpha=0.2)
    plt.savefig(os.path.join(result_path, 'acc_' + file_name + '.png'))
    plt.close(plt_acc)

    #損失値のプロット
    plt_loss = plt.figure()
    plt.plot(epochs,smooth_curve(loss),"go",label="Training Loss")
    plt.plot(epochs,smooth_curve(val_loss),"bo",label="Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(color='gray', alpha=0.2)
    plt.savefig(os.path.join(result_path, 'loss_' + file_name + '.png'))
    plt.close(plt_loss)

    #混合行列


    #モデルの評価
    evaluate_value = model.evaluate_generator(test_generator)
    print('test acc_original :', '{:.1%}'.format(evaluate_value[1]))

    print('Learning by Original model finished!')

    #predict model and display images
    files=os.listdir(display_dir)

    n_display = min(49, len(files))
    img=random.sample(files,n_display)

    plt.figure(figsize=(10,10))

    for i in range(n_display):
        test_img_path     = os.path.join(display_dir,img[i])
        test_img_name     = os.path.basename(test_img_path)
        test_img_onlyname = os.path.splitext(test_img_name)[0]
        temp_img          = load_img(test_img_path,target_size=(input_image_size,input_image_size))
        plt.subplot(5,7,i+1)
        plt.imshow(temp_img)
        #Images normalization
        temp_img_array=img_to_array(temp_img)
        temp_img_array=temp_img_array.astype('float32')/255.0
        temp_img_array=temp_img_array.reshape((1,input_image_size,input_image_size,3))
        #predict image
        img_pred=model.predict(temp_img_array)
        #Graph setting
        plt.title(label[np.argmax(img_pred)] + str(round(max(img_pred[0]),2)))
        plt.xlabel(test_img_onlyname)
        plt.xticks([]),plt.yticks([])

    print('finished!')
    plt.show()




def choose_folder():
    print('学習させたいデータベースのフォルダーを選んでください')
    print('そのフォルダー内に　train, test, validation, display　フォルダが必要です。')

    data_folder_path = tkinter.filedialog.askdirectory(initialdir = data_processed_path,
                        title = 'choose data folder')

    t_foldername.set(str(Path(data_folder_path).name))
    t_folderpath.set(data_folder_path)

    t_theme_name.set(Path(data_folder_path).name)

    return


# tkinter
root = tkinter.Tk()
#root.withdraw()

font1 = font.Font(family='游ゴシック', size=10, weight='bold')
root.option_add("*Font", font1)
style1  = ttk.Style()
style1.configure('my.TButton', font = ('游ゴシック',10)  )
root.title('Deep learning for photos')

frame1  = tkinter.ttk.Frame(root, height = 500, width = 500)
frame1.grid(row=0,column=0,sticky=(N,E,S,W))

label_folder  = tkinter.ttk.Label(frame1, text = 'フォルダパス:', anchor="w")

t_foldername = tkinter.StringVar()
t_folderpath = tkinter.StringVar()

entry_foldername  = ttk.Entry(frame1, textvariable = t_foldername, width = 20)

button_choose_folder    = ttk.Button(frame1, text='フォルダ選択',
                                 command = choose_folder, style = 'my.TButton')


#label_theme_name  = tkinter.ttk.Label(frame1, text = 'テーマ名を入力:')
t_theme_name = tkinter.StringVar()
#entry_theme_name  = ttk.Entry(frame1, textvariable = t_theme_name, width = 20)

#save_folder_name = os.path.dirname(t_foldername.get()) + 'result'


var_epoch = tkinter.StringVar()
var_epoch.set('low')

Radiobutton_epoch_high = tkinter.Radiobutton(frame1, value = 'high',   text = 'とことん', variable = var_epoch)
Radiobutton_epoch_middle = tkinter.Radiobutton(frame1, value = 'middle', text = '普通', variable = var_epoch)
Radiobutton_epoch_low = tkinter.Radiobutton(frame1, value = 'low',    text = '軽く', variable = var_epoch)

label_epoch =tkinter.ttk.Label(frame1, text = 'エポック数:', anchor="w")

var_model = tkinter.StringVar()
var_model.set('mobilenet')

Radiobutton_epoch_Inception = tkinter.Radiobutton(frame1, value = 'Inception',   text = 'Inception', variable = var_model)
Radiobutton_epoch_xception = tkinter.Radiobutton(frame1, value = 'xception',   text = 'xception', variable = var_model)
Radiobutton_epoch_mobilenet = tkinter.Radiobutton(frame1, value = 'mobilenet', text = 'mobilenet', variable = var_model)
Radiobutton_epoch_vgg16 = tkinter.Radiobutton(frame1, value = 'vgg16',    text = 'vgg16', variable = var_model)
Radiobutton_epoch_original = tkinter.Radiobutton(frame1, value = 'original',    text = 'original', variable = var_model)

label_model =tkinter.ttk.Label(frame1, text = 'モデル:', anchor="w")

Booleanvar_savemodel = tkinter.BooleanVar()
Booleanvar_savemodel.set(False)
Checkbutton_savemodel = tkinter.Checkbutton(frame1, text = 'モデル保存', variable = Booleanvar_savemodel)
button_learning     = ttk.Button(frame1, text='訓練開始',
                                 command = main, style = 'my.TButton')

button_choose_folder.grid(row=1,column=2,sticky=W)
label_folder.grid(row=2,column=1,sticky=E)
entry_foldername.grid(row=2,column=2,sticky=W)

label_epoch.grid(row =3, column =1,sticky=E)
Radiobutton_epoch_high.grid(row =3, column =2,sticky=W)
Radiobutton_epoch_middle.grid(row =3, column =3,sticky=W)
Radiobutton_epoch_low.grid(row =3, column =4,sticky=W)

label_model.grid(row =4, column =1,sticky=E)
Radiobutton_epoch_Inception.grid(row=4,column=2,sticky=W)
Radiobutton_epoch_xception.grid(row=4,column=3,sticky=W)
Radiobutton_epoch_mobilenet.grid(row=4,column=4,sticky=W)
Radiobutton_epoch_vgg16.grid(row=4,column=5,sticky=W)
Radiobutton_epoch_original.grid(row=4,column=6,sticky=W)

#label_theme_name.grid(row=4,column=1,sticky=E)
#entry_theme_name.grid(row=4,column=2,sticky=W)

Checkbutton_savemodel.grid(row = 7, column =2, sticky = W)
button_learning.grid(row = 10 , column = 2, sticky = W)

for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)
root.mainloop()
