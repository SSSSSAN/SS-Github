import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras.models
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

input_image_size     = 224
model                = load_model('test.h5')

program_path         = Path(__file__).parent.resolve() #src
parent_path          = program_path.parent.resolve()   #deeplearning_for_photos
data_path            = parent_path / 'data'
data_processed_path  = data_path / 'processed'
predict_dir          = os.path.join(data_processed_path, 'test')

files                = os.listdir(predict_dir)
n_files              = len(files)

labels               = ['A','B','C','D','E','F']

for i in range(n_files):
    predict_img_path     = os.path.join(predict_dir, files[i])
    predict_img_name     = os.path.basename(predict_img_path)
    predict_img_onlyname = os.path.splitext(predict_img_name)[0]
    temp_img             = load_img(predict_img_path,target_size=(input_image_size,input_image_size))

    #Images normalization
    temp_img_array       = img_to_array(temp_img)
    temp_img_array       = temp_img_array.astype('float32')/255.0
    temp_img_array       = temp_img_array.reshape((1,input_image_size,input_image_size,3))
    #predict results
    img_pred             = model.predict(temp_img_array)
    label_pred           = labels[np.argmax(img_pred)]
    
    #print('finished!')
    #print(files)
    print('画像名：' + predict_img_onlyname + ', 予想ラベル名:' + label_pred +':' + str(round(max(img_pred[0]),2)))

print(label_pred.count('A'))
print(label_pred.count('B'))
print(label_pred.count('C'))
print(label_pred.count('D'))
print(label_pred.count('E'))
print(label_pred.count('F'))

#print(sum(label_pred.count('A')))
#result_A_df = pd.DataFrame({label_pred.count('A')}, columns='A')
#pred_results_df = pd.DataFrame(labels[np.argmax(img_pred)], str(round(max(img_pred[0]),2)))
#pred_results_df.to_csv(parent_path, index=False)

print('finished!')

#https://qiita.com/takubb/items/d449e7760796287080c8
'''
img_width, img_height = 224, 224

model = load_model('test.h5')

img_predict = []
for image_name in os.listdir(test_dir):
    try:
        img = Image.open(os.path.join(test_dir, image_name))
        img = img.convert("RGB")
        img = img.resize((img_width, img_height))
        img_array = np.asarray(img)
        img_predict.append(img_array)
    except Exception as e:
        pass

img_predict = np.asarray(img_predict)
print('img_predict.shape = ', img_predict.shape)

result_predict = model.predict(img_predict)

for i in range(len(img_predict)):
  print("predict=%s, " % (result_predict[0]))
'''
'''
program_path        = Path(__file__).parent.resolve() #src
parent_path         = program_path.parent.resolve()   #deeplearning_for_photos
data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'
test_dir            = os.path.join(data_processed_path, 'test')

test_datagen=ImageDataGenerator(rescale=1.0/255)
test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)

model      = load_model(test.h5)
test_score = model.predict(test_generator)
print('\n test loss:',test_score[0])
print('\n test_acc:',test_score[1])


print('finished!')
'''
