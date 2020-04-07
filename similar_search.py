#VGG19中のfc2層までの特徴量抽出法を使用し、任意のフォルダの画像の中から目的の画像と類似する画像を探す


import numpy as np
import glob
from pathlib import Path
import tkinter
import tkinter.filedialog
from tkinter import ttk,N,E,S,W,font

#https://github.com/nmslib/nmslib/blob/master/README.md
import nmslib

import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)


#Pathの設定
current_path = Path.cwd()
program_path = Path(__file__).parent.resolve()
parent_path = program_path.parent.resolve()

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'

def main():
    print("データベースを選択してください")
    print("サブディレクトリ内の画像もすべて検索対象となります")

    data_folder_path = tkinter.filedialog.askdirectory(initialdir = data_processed_path,
                        title = 'choose data folder')

    print("データベースと比較したい画像を選択してください")
    test_img_path = tkinter.filedialog.askopenfilename(initialdir = data_processed_path,
                        title = 'choose test image', filetypes = [('image file', '*.jpeg;*jpg;*png')])

    import keras
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.applications.inception_v3 import InceptionV3
    from keras.preprocessing import image
    from keras.applications.vgg19 import preprocess_input

    # refer https://qiita.com/K-jun/items/cab923d49a939a8486fc
    from keras.applications.vgg19 import VGG19, preprocess_input

    base_model = VGG19(weights="imagenet")
    #base_model.summary()
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

    test_img = image.load_img(test_img_path, target_size=(224, 224))
    x = image.img_to_array(test_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    test_fc2_features = model.predict(x)

    png_list  = glob.glob(data_folder_path + "/**/*.png", recursive=True)
    jpeg_list = glob.glob(data_folder_path + "/**/*.jpeg", recursive=True)
    jpg_list  = glob.glob(data_folder_path + "/**/*.jpg", recursive=True)
    image_list = png_list + jpeg_list + jpg_list

    fc2_list = []
    print("画像数は" , len(image_list), "です。")

    print("探索を開始します。中止する場合はCtrl+Cを押してください")
    import time
    start_time = time.time()

    warnings.resetwarnings()
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', FutureWarning)

    for i, image_path in enumerate(image_list):
        if  (i == 10) or (i != 0 and i % 100 == 0)  :
            remained_num =  len(image_list) - i
            elapsed_time = time.time()
            remained_time = (elapsed_time-start_time)/i *  remained_num
            print(i, "件完了しました。　残り", remained_num , "件、" , round(remained_time) ,"秒くらいかかります。")
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc2_features = model.predict(x)
        fc2_list.append(fc2_features[0])

    warnings.resetwarnings()
    #print(fc2_list)

    # refer https://qiita.com/wasnot/items/20c4f30a529ae3ed5f52
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(fc2_list)
    index.createIndex({'post': 2}, print_progress=True)

    ids, distances = index.knnQuery(test_fc2_features, k=len(image_list))
    result = [image_list[i] for i in ids]
    print(ids)
    print(distances)
    print(result)

    print("選択した画像は " , test_img_path, " です")

    print("選択した画像に似ている順に表示します")
    for i, id in enumerate(ids):
        print(image_list[id], " : 距離： ", distances[i])

    print("選択した画像は " , test_img_path, " です")

    #index.save(model_name)
    import time
    print("30分後に画面を閉じます")
    time.sleep(1800)


if __name__ == "__main__" :
    main()



'''
# tkinter
root = tkinter.Tk()

font1 = font.Font(family='游ゴシック', size=10, weight='bold')
root.option_add("*Font", font1)
style1  = ttk.Style()
style1.configure('my.TButton', font = ('游ゴシック',10)  )

root.title('Find the similar images')

frame1  = tkinter.ttk.Frame(root, height = 500, width = 500)
frame1.grid(row=0,column=0,sticky=(N,E,S,W))



button_start  = ttk.Button(frame1, text='データフォルダ選択',
                                 command = main, style = 'my.TButton')

#label_button  = tkinter.ttk.Label(frame1, text = 'モデル保存:', anchor="w")



for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)
root.mainloop()
'''
