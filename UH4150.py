#分光データ(txt)をファイル一覧から取得し、1つずつグラフを作成する

import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt
import tkinter
import tkinter.filedialog

#ファイルダイアログを開いてGUIで操作できるライブラリ
tk = tkinter.Tk()
#ファイルダイアログを表示
tk.withdraw()
#カレントディレクトリの取得
currentdirectory = os.getcwd()
print('ファイルを選択してください')
#ファイル選択ダイアログの表示
txtfile_path = tkinter.filedialog.askopenfilename(initialdir = currentdirectory,title = 'txtファイルを選択してください')
#txtfile_pathという変数にtxtfile_pathのフォルダ名を入れる
#txtfile_path の中身：'C:\Users\1710099\Desktop\*.txt'
txtfolder_path = os.path.dirname(txtfile_path)
#フォルダを移動
#C:\Users\1310202\Desktop からC:\Users\1710099\Desktop\*.txtへ移る
os.chdir(txtfolder_path)
#globを使ってファイル一覧を取得
filelist = glob.glob(txtfolder_path+'/*.txt')

for filename in filelist:

    with open(filename) as f:
        # ﾃﾞｰﾀﾘｽﾄまでの行数をカウントする
        nSkiprow = 0
        for line in f.readlines():
            nSkiprow += 1
            if line.startswith("ﾃﾞｰﾀﾘｽﾄ"):
                #ﾃﾞｰﾀﾘｽﾄが見つかったらforを終了する
                break

        # 上で求めた行数をスキップしてread_tableで読み込む
        df = pd.read_table(filename, skiprows=nSkiprow, encoding='shift_jis')

        #グラフの作成
        plt.plot(df["nm"],df["%T"],marker="o")
        #グラフの軸
        plt.xlabel("Wavelength[nm]")
        plt.ylabel("Transmittance[%]")
        #グラフ保存
        plt.savefig(filename+'.png')

sys.exit()






#親データフレーム
x=df[0]
y=df[1]

#グラフ
fig = plt.figure()

#複数グラフ設定
#ax1 = fig.add_subplot(111) #Graph1

#x軸のスケール設定
ax1.set_xscale('nm')
#y軸のスケール設定
ax1.set_yscale('%T')

#軸の範囲
ax1.set_xlim(300, 2500)
ax1.set_ylim(0,100)


#個別タイトル
ax1.set_title("Graph1",fontdict = {"ITO": fp},fontsize=12)

#軸
ax1.set_xlabel("x",fontdict = {"ITO": fp},fontsize=12)
ax1.set_ylabel("y",fontdict = {"ITO": fp},fontsize=12)

#プロット
ax1.plot(x, y,'blue',label='graph1')

#Legend 位置
ax1.legend(loc="upper right")


#レイアウト調整
plt.tight_layout()

#グラフ全体のタイトル
fig.suptitle('Graph', fontsize=14)

plt.subplots_adjust(top=0.85)

#ファイルを保存 pngで保存
plt.savefig("sample.png")

plt.show()

import sys
sys.exit()
