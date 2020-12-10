import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt

"""
フォルダの読み込み
for文でフォルダ内の1つずつのファイルを指定
　txtからcsvへ変換、指定フォルダに格納
指定されたフォルダ内のcsvファイルの%列を結合して1つのファイルに出力
出力されたファイルを主成分分析
"""

with open("18100401_0tr(UDS).txt") as f:
    # ﾃﾞｰﾀﾘｽﾄまでの行数をカウントする
    nSkiprow = 0
    for line in f.readlines():
        nSkiprow += 1
        if line.startswith("ﾃﾞｰﾀﾘｽﾄ"):
            #ﾃﾞｰﾀﾘｽﾄが見つかったらforを終了する
            break

    # 上で求めた行数をスキップしてread_tableで読み込む
    df = pd.read_table("18100401_0tr(UDS).txt", skiprows=nSkiprow, encoding='shift_jis')
    
    #csv書き出し
    df.to_csv('18100401_0tr(UDS).csv')
