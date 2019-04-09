#Kinectから得られたCSVをSVMの学習用に使えるデータに変換する
#SVMプログラム用のベクトルデータに変換
#Color...のカラムを抜いてベクトルの値とその距離にする
"""
datas --- 1 --- body.csv
            --- lines.txt
      --- 2 ---...
"""

import pandas as pd
import numpy as np
import math
import datetime
import os#ファイル作成用
import shutil#ファイルコピー用
import line_splitter as ls#ベクトル化用プログラム
import glob#ファイル検索用

#分割するフォルダの配列
root = glob.glob("./test/*")
#print(root)

for f in root:
    #print(f)
    csvfile = f + "\\body.csv"
    linesfile = f + "\\lines.txt"

    if os.path.exists(csvfile):
        print(csvfile)
        print(linesfile)

        #データフレーム型に変換
        df = pd.read_csv(csvfile, header=1)

        #行と列の数
        #print(df.shape)
        #列のヘッダー行列
        #print(df.columns)
        #二行目を取り出す（カラム配列の二番目を取り出す）
        #print(df[df.columns[1]])

        #全部のカラムのインデックス
        indexes = df.columns

        #Colorとつくカラムを抜いたインデックス
        #indexesの中からColorとつかないものをsとして取り出し
        #noColorIndexes = [s for s in indexes if 'Color' not in s]


        ###いらない座標を消す
        #Colorとつくカラムのインデックス
        ColorIndexes = [s for s in indexes if 'Color' in s]
        #print(ColorIndexes)
        #Colorとつく列を消す
        df = df.drop(columns = ColorIndexes)
        #最後のいらない部分を消す
        df = df.drop(columns = 'Unnamed: 126')


        ###時間を変換する
        #最初のカラムをdatetime型にする
        try:
            times = pd.to_datetime(df[df.columns[0]], format='%H:%M:%S.%f')
        except:#方に合わないときは大体自動で変換できる
            times = pd.to_datetime(df[df.columns[0]])
        #ミリ秒基準にする
        allMicroSeconds = [int(t.hour*3600000 + t.minute*60000 + t.second*1000 + t.microsecond/1000) for t in times]
        #時間のカラムに上書き
        df[df.columns[0]] = allMicroSeconds

        ###ベクトルデータの作成

        #SPINE_SHOULDER -> HEAD
        SP_SHOtoHEAD = pd.DataFrame()
        SP_SHOtoHEAD = SP_SHOtoHEAD.assign(
            CameraX = df['CameraX.3'] - df['CameraX.20']
        )
        SP_SHOtoHEAD = SP_SHOtoHEAD.assign(
            CameraY = df['CameraY.3'] - df['CameraY.20']
        )
        SP_SHOtoHEAD = SP_SHOtoHEAD.assign(
            CameraZ = df['CameraZ.3'] - df['CameraZ.20']
        )
        SP_SHOtoHEAD = SP_SHOtoHEAD.assign(
            Norm = np.sqrt(SP_SHOtoHEAD['CameraX'] **2
                        + SP_SHOtoHEAD['CameraY'] **2
                        + SP_SHOtoHEAD['CameraZ'] **2)
        )
        SP_SHOtoHEAD['CameraX'] = SP_SHOtoHEAD['CameraX'] / SP_SHOtoHEAD['Norm']
        SP_SHOtoHEAD['CameraY'] = SP_SHOtoHEAD['CameraY'] / SP_SHOtoHEAD['Norm']
        SP_SHOtoHEAD['CameraZ'] = SP_SHOtoHEAD['CameraZ'] / SP_SHOtoHEAD['Norm']
        #print(SP_SHOtoHEAD)

        #SPINE_SHOULDER -> SHOULDER_RIGHT
        SP_SHOtoSHO_R = pd.DataFrame()
        SP_SHOtoSHO_R= SP_SHOtoSHO_R.assign(
            CameraX = df['CameraX.8'] - df['CameraX.20']
        )
        SP_SHOtoSHO_R = SP_SHOtoSHO_R.assign(
            CameraY = df['CameraY.8'] - df['CameraY.20']
        )
        SP_SHOtoSHO_R = SP_SHOtoSHO_R.assign(
            CameraZ = df['CameraZ.8'] - df['CameraZ.20']
        )
        SP_SHOtoSHO_R = SP_SHOtoSHO_R.assign(
            Norm = np.sqrt(SP_SHOtoSHO_R['CameraX'] **2
                        + SP_SHOtoSHO_R['CameraY'] **2
                        + SP_SHOtoSHO_R['CameraZ'] **2)
        )
        SP_SHOtoSHO_R['CameraX'] = SP_SHOtoSHO_R['CameraX'] / SP_SHOtoSHO_R['Norm']
        SP_SHOtoSHO_R['CameraY'] = SP_SHOtoSHO_R['CameraY'] / SP_SHOtoSHO_R['Norm']
        SP_SHOtoSHO_R['CameraZ'] = SP_SHOtoSHO_R['CameraZ'] / SP_SHOtoSHO_R['Norm']
        #print(SP_SHOtoSHO_R)

        #SPINE_SHOULDER -> SHOULDER_LEFT
        SP_SHOtoSHO_L = pd.DataFrame()
        SP_SHOtoSHO_L = SP_SHOtoSHO_L.assign(
            CameraX = df['CameraX.4'] - df['CameraX.20']
        )
        SP_SHOtoSHO_L = SP_SHOtoSHO_L.assign(
            CameraY = df['CameraY.4'] - df['CameraY.20']
        )
        SP_SHOtoSHO_L = SP_SHOtoSHO_L.assign(
            CameraZ = df['CameraZ.4'] - df['CameraZ.20']
        )
        SP_SHOtoSHO_L = SP_SHOtoSHO_L.assign(
            Norm = np.sqrt(SP_SHOtoSHO_L['CameraX'] **2
                        + SP_SHOtoSHO_L['CameraY'] **2
                        + SP_SHOtoSHO_L['CameraZ'] **2)
        )
        SP_SHOtoSHO_L['CameraX'] = SP_SHOtoSHO_L['CameraX'] / SP_SHOtoSHO_L['Norm']
        SP_SHOtoSHO_L['CameraY'] = SP_SHOtoSHO_L['CameraY'] / SP_SHOtoSHO_L['Norm']
        SP_SHOtoSHO_L['CameraZ'] = SP_SHOtoSHO_L['CameraZ'] / SP_SHOtoSHO_L['Norm']
        #print(SP_SHOtoSHO_L)

        #SPINE_SHOULDER -> SPINE_MID
        SP_SHOtoSP_MID = pd.DataFrame()
        SP_SHOtoSP_MID = SP_SHOtoSP_MID.assign(
            CameraX = df['CameraX.1'] - df['CameraX.20']
        )
        SP_SHOtoSP_MID = SP_SHOtoSP_MID.assign(
            CameraY = df['CameraY.1'] - df['CameraY.20']
        )
        SP_SHOtoSP_MID = SP_SHOtoSP_MID.assign(
            CameraZ = df['CameraZ.1'] - df['CameraZ.20']
        )
        SP_SHOtoSP_MID = SP_SHOtoSP_MID.assign(
            Norm = np.sqrt(SP_SHOtoSP_MID['CameraX'] **2
                        + SP_SHOtoSP_MID['CameraY'] **2
                        + SP_SHOtoSP_MID['CameraZ'] **2)
        )
        SP_SHOtoSP_MID['CameraX'] = SP_SHOtoSP_MID['CameraX'] / SP_SHOtoSP_MID['Norm']
        SP_SHOtoSP_MID['CameraY'] = SP_SHOtoSP_MID['CameraY'] / SP_SHOtoSP_MID['Norm']
        SP_SHOtoSP_MID['CameraZ'] = SP_SHOtoSP_MID['CameraZ'] / SP_SHOtoSP_MID['Norm']
        #print(SP_SHOtoSP_MID)

        #SHOULDER_RIGHT -> ELBOW_RIGHT
        SHO_RtoELB_R = pd.DataFrame()
        SHO_RtoELB_R = SHO_RtoELB_R.assign(
            CameraX = df['CameraX.9'] - df['CameraX.8']
        )
        SHO_RtoELB_R = SHO_RtoELB_R.assign(
            CameraY = df['CameraY.9'] - df['CameraY.8']
        )
        SHO_RtoELB_R = SHO_RtoELB_R.assign(
            CameraZ = df['CameraZ.9'] - df['CameraZ.8']
        )
        SHO_RtoELB_R = SHO_RtoELB_R.assign(
            Norm = np.sqrt(SHO_RtoELB_R['CameraX'] **2
                        + SHO_RtoELB_R['CameraY'] **2
                        + SHO_RtoELB_R['CameraZ'] **2)
        )
        SHO_RtoELB_R['CameraX'] = SHO_RtoELB_R['CameraX'] / SHO_RtoELB_R['Norm']
        SHO_RtoELB_R['CameraY'] = SHO_RtoELB_R['CameraY'] / SHO_RtoELB_R['Norm']
        SHO_RtoELB_R['CameraZ'] = SHO_RtoELB_R['CameraZ'] / SHO_RtoELB_R['Norm']
        #print(SHO_RtoELB_R)

        #ELBOW_RIGHT -> HAND_RIGHT
        ELB_RtoHAN_R = pd.DataFrame()
        ELB_RtoHAN_R = ELB_RtoHAN_R.assign(
            CameraX = df['CameraX.11'] - df['CameraX.9']
        )
        ELB_RtoHAN_R = ELB_RtoHAN_R.assign(
            CameraY = df['CameraY.11'] - df['CameraY.9']
        )
        ELB_RtoHAN_R = ELB_RtoHAN_R.assign(
            CameraZ = df['CameraZ.11'] - df['CameraZ.9']
        )
        ELB_RtoHAN_R = ELB_RtoHAN_R.assign(
            Norm = np.sqrt(ELB_RtoHAN_R['CameraX'] **2
                        + ELB_RtoHAN_R['CameraY'] **2
                        + ELB_RtoHAN_R['CameraZ'] **2)
        )
        ELB_RtoHAN_R['CameraX'] = ELB_RtoHAN_R['CameraX'] / ELB_RtoHAN_R['Norm']
        ELB_RtoHAN_R['CameraY'] = ELB_RtoHAN_R['CameraY'] / ELB_RtoHAN_R['Norm']
        ELB_RtoHAN_R['CameraZ'] = ELB_RtoHAN_R['CameraZ'] / ELB_RtoHAN_R['Norm']
        #print(ELB_RtoHAN_R)

        #SHOULDER_LEFT -> ELBOW_LEFT
        SHO_LtoELB_L = pd.DataFrame()
        SHO_LtoELB_L = SHO_LtoELB_L.assign(
            CameraX = df['CameraX.5'] - df['CameraX.4']
        )
        SHO_LtoELB_L = SHO_LtoELB_L.assign(
            CameraY = df['CameraY.5'] - df['CameraY.4']
        )
        SHO_LtoELB_L = SHO_LtoELB_L.assign(
            CameraZ = df['CameraZ.5'] - df['CameraZ.4']
        )
        SHO_LtoELB_L = SHO_LtoELB_L.assign(
            Norm = np.sqrt(SHO_LtoELB_L['CameraX'] **2
                        + SHO_LtoELB_L['CameraY'] **2
                        + SHO_LtoELB_L['CameraZ'] **2)
        )
        SHO_LtoELB_L['CameraX'] = SHO_LtoELB_L['CameraX'] / SHO_LtoELB_L['Norm']
        SHO_LtoELB_L['CameraY'] = SHO_LtoELB_L['CameraY'] / SHO_LtoELB_L['Norm']
        SHO_LtoELB_L['CameraZ'] = SHO_LtoELB_L['CameraZ'] / SHO_LtoELB_L['Norm']
        #print(SHO_LtoELB_L)

        #ELBOW_LEFT -> HAND_LEFT
        ELB_LtoHAN_L = pd.DataFrame()
        ELB_LtoHAN_L = ELB_LtoHAN_L.assign(
            CameraX = df['CameraX.7'] - df['CameraX.5']
        )
        ELB_LtoHAN_L = ELB_LtoHAN_L.assign(
            CameraY = df['CameraY.7'] - df['CameraY.5']
        )
        ELB_LtoHAN_L = ELB_LtoHAN_L.assign(
            CameraZ = df['CameraZ.7'] - df['CameraZ.5']
        )
        ELB_LtoHAN_L = ELB_LtoHAN_L.assign(
            Norm = np.sqrt(ELB_LtoHAN_L['CameraX'] **2
                        + ELB_LtoHAN_L['CameraY'] **2
                        + ELB_LtoHAN_L['CameraZ'] **2)
        )
        ELB_LtoHAN_L['CameraX'] = ELB_LtoHAN_L['CameraX'] / ELB_LtoHAN_L['Norm']
        ELB_LtoHAN_L['CameraY'] = ELB_LtoHAN_L['CameraY'] / ELB_LtoHAN_L['Norm']
        ELB_LtoHAN_L['CameraZ'] = ELB_LtoHAN_L['CameraZ'] / ELB_LtoHAN_L['Norm']
        #print(ELB_LtoHAN_L)

        #SPINE_MID -> SPINE_BASE
        SP_MIDtoSP_BASE = pd.DataFrame()
        SP_MIDtoSP_BASE = SP_MIDtoSP_BASE.assign(
            CameraX = df['CameraX'] - df['CameraX.1']
        )
        SP_MIDtoSP_BASE = SP_MIDtoSP_BASE.assign(
            CameraY = df['CameraY'] - df['CameraY.1']
        )
        SP_MIDtoSP_BASE = SP_MIDtoSP_BASE.assign(
            CameraZ = df['CameraZ'] - df['CameraZ.1']
        )
        SP_MIDtoSP_BASE = SP_MIDtoSP_BASE.assign(
            Norm = np.sqrt(SP_MIDtoSP_BASE['CameraX'] **2
                        + SP_MIDtoSP_BASE['CameraY'] **2
                        + SP_MIDtoSP_BASE['CameraZ'] **2)
        )
        ELB_LtoHAN_L['CameraX'] = ELB_LtoHAN_L['CameraX'] / ELB_LtoHAN_L['Norm']
        ELB_LtoHAN_L['CameraY'] = ELB_LtoHAN_L['CameraY'] / ELB_LtoHAN_L['Norm']
        ELB_LtoHAN_L['CameraZ'] = ELB_LtoHAN_L['CameraZ'] / ELB_LtoHAN_L['Norm']
        #print(SP_MIDtoSP_BASE)

        #SPINE_BASE -> HIP_RIGHT
        SP_BASEtoHIP_R = pd.DataFrame()
        SP_BASEtoHIP_R = SP_BASEtoHIP_R.assign(
            CameraX = df['CameraX.16'] - df['CameraX']
        )
        SP_BASEtoHIP_R = SP_BASEtoHIP_R.assign(
            CameraY = df['CameraY.16'] - df['CameraY']
        )
        SP_BASEtoHIP_R = SP_BASEtoHIP_R.assign(
            CameraZ = df['CameraZ.16'] - df['CameraZ']
        )
        SP_BASEtoHIP_R = SP_BASEtoHIP_R.assign(
            Norm = np.sqrt(SP_BASEtoHIP_R['CameraX'] **2
                        + SP_BASEtoHIP_R['CameraY'] **2
                        + SP_BASEtoHIP_R['CameraZ'] **2)
        )
        SP_BASEtoHIP_R['CameraX'] = SP_BASEtoHIP_R['CameraX'] / SP_BASEtoHIP_R['Norm']
        SP_BASEtoHIP_R['CameraY'] = SP_BASEtoHIP_R['CameraY'] / SP_BASEtoHIP_R['Norm']
        SP_BASEtoHIP_R['CameraZ'] = SP_BASEtoHIP_R['CameraZ'] / SP_BASEtoHIP_R['Norm']
        #print(SP_BASEtoHIP_R)

        #SPINE_BASE -> HIP_LEFT
        SP_BASEtoHIP_L = pd.DataFrame()
        SP_BASEtoHIP_L = SP_BASEtoHIP_L.assign(
            CameraX = df['CameraX.12'] - df['CameraX']
        )
        SP_BASEtoHIP_L = SP_BASEtoHIP_L.assign(
            CameraY = df['CameraY.12'] - df['CameraY']
        )
        SP_BASEtoHIP_L = SP_BASEtoHIP_L.assign(
            CameraZ = df['CameraZ.12'] - df['CameraZ']
        )
        SP_BASEtoHIP_L = SP_BASEtoHIP_L.assign(
            Norm = np.sqrt(SP_BASEtoHIP_L['CameraX'] **2
                        + SP_BASEtoHIP_L['CameraY'] **2
                        + SP_BASEtoHIP_L['CameraZ'] **2)
        )
        SP_BASEtoHIP_L['CameraX'] = SP_BASEtoHIP_L['CameraX'] / SP_BASEtoHIP_L['Norm']
        SP_BASEtoHIP_L['CameraY'] = SP_BASEtoHIP_L['CameraY'] / SP_BASEtoHIP_L['Norm']
        SP_BASEtoHIP_L['CameraZ'] = SP_BASEtoHIP_L['CameraZ'] / SP_BASEtoHIP_L['Norm']
        #print(SP_BASEtoHIP_L)

        #HIP_RIGHT -> KNEE_RIGHT
        HIP_RtoKNE_R = pd.DataFrame()
        HIP_RtoKNE_R = HIP_RtoKNE_R.assign(
            CameraX = df['CameraX.17'] - df['CameraX.16']
        )
        HIP_RtoKNE_R = HIP_RtoKNE_R.assign(
            CameraY = df['CameraY.17'] - df['CameraY.16']
        )
        HIP_RtoKNE_R = HIP_RtoKNE_R.assign(
            CameraZ = df['CameraZ.17'] - df['CameraZ.16']
        )
        HIP_RtoKNE_R = HIP_RtoKNE_R.assign(
            Norm = np.sqrt(HIP_RtoKNE_R['CameraX'] **2
                        + HIP_RtoKNE_R['CameraY'] **2
                        + HIP_RtoKNE_R['CameraZ'] **2)
        )
        HIP_RtoKNE_R['CameraX'] = HIP_RtoKNE_R['CameraX'] / HIP_RtoKNE_R['Norm']
        HIP_RtoKNE_R['CameraY'] = HIP_RtoKNE_R['CameraY'] / HIP_RtoKNE_R['Norm']
        HIP_RtoKNE_R['CameraZ'] = HIP_RtoKNE_R['CameraZ'] / HIP_RtoKNE_R['Norm']
        #print(HIP_RtoKNE_R)

        #KNEE_RIGHT -> ANKLE_RIGHT
        KNE_RtoANK_R = pd.DataFrame()
        KNE_RtoANK_R = KNE_RtoANK_R.assign(
            CameraX = df['CameraX.18'] - df['CameraX.17']
        )
        KNE_RtoANK_R = KNE_RtoANK_R.assign(
            CameraY = df['CameraY.18'] - df['CameraY.17']
        )
        KNE_RtoANK_R = KNE_RtoANK_R.assign(
            CameraZ = df['CameraZ.18'] - df['CameraZ.17']
        )
        KNE_RtoANK_R = KNE_RtoANK_R.assign(
            Norm = np.sqrt(KNE_RtoANK_R['CameraX'] **2
                        + KNE_RtoANK_R['CameraY'] **2
                        + KNE_RtoANK_R['CameraZ'] **2)
        )
        KNE_RtoANK_R['CameraX'] = KNE_RtoANK_R['CameraX'] / KNE_RtoANK_R['Norm']
        KNE_RtoANK_R['CameraY'] = KNE_RtoANK_R['CameraY'] / KNE_RtoANK_R['Norm']
        KNE_RtoANK_R['CameraZ'] = KNE_RtoANK_R['CameraZ'] / KNE_RtoANK_R['Norm']
        #print(KNE_RtoANK_R)

        #ANKLE_RIGHT -> FOOT_RIGHT
        ANK_RtoFOO_R = pd.DataFrame()
        ANK_RtoFOO_R = ANK_RtoFOO_R.assign(
            CameraX = df['CameraX.19'] - df['CameraX.18']
        )
        ANK_RtoFOO_R = ANK_RtoFOO_R.assign(
            CameraY = df['CameraY.19'] - df['CameraY.18']
        )
        ANK_RtoFOO_R = ANK_RtoFOO_R.assign(
            CameraZ = df['CameraZ.19'] - df['CameraZ.18']
        )
        ANK_RtoFOO_R = ANK_RtoFOO_R.assign(
            Norm = np.sqrt(ANK_RtoFOO_R['CameraX'] **2
                        + ANK_RtoFOO_R['CameraY'] **2
                        + ANK_RtoFOO_R['CameraZ'] **2)
        )
        ANK_RtoFOO_R['CameraX'] = ANK_RtoFOO_R['CameraX'] / ANK_RtoFOO_R['Norm']
        ANK_RtoFOO_R['CameraY'] = ANK_RtoFOO_R['CameraY'] / ANK_RtoFOO_R['Norm']
        ANK_RtoFOO_R['CameraZ'] = ANK_RtoFOO_R['CameraZ'] / ANK_RtoFOO_R['Norm']
        #print(ANK_RtoFOO_R)

        #HIP_LEFT -> KNEE_LEFT
        HIP_LtoKNE_L = pd.DataFrame()
        HIP_LtoKNE_L = HIP_LtoKNE_L.assign(
            CameraX = df['CameraX.13'] - df['CameraX.12']
        )
        HIP_LtoKNE_L = HIP_LtoKNE_L.assign(
            CameraY = df['CameraY.13'] - df['CameraY.12']
        )
        HIP_LtoKNE_L = HIP_LtoKNE_L.assign(
            CameraZ = df['CameraZ.13'] - df['CameraZ.12']
        )
        HIP_LtoKNE_L = HIP_LtoKNE_L.assign(
            Norm = np.sqrt(HIP_LtoKNE_L['CameraX'] **2
                        + HIP_LtoKNE_L['CameraY'] **2
                        + HIP_LtoKNE_L['CameraZ'] **2)
        )
        HIP_LtoKNE_L['CameraX'] = HIP_LtoKNE_L['CameraX'] / HIP_LtoKNE_L['Norm']
        HIP_LtoKNE_L['CameraY'] = HIP_LtoKNE_L['CameraY'] / HIP_LtoKNE_L['Norm']
        HIP_LtoKNE_L['CameraZ'] = HIP_LtoKNE_L['CameraZ'] / HIP_LtoKNE_L['Norm']
        #print(HIP_LtoKNE_L)

        #KNEE_LEFT -> ANKLE_LEFT
        KNE_LtoANK_L = pd.DataFrame()
        KNE_LtoANK_L = KNE_LtoANK_L.assign(
            CameraX = df['CameraX.14'] - df['CameraX.13']
        )
        KNE_LtoANK_L = KNE_LtoANK_L.assign(
            CameraY = df['CameraY.14'] - df['CameraY.13']
        )
        KNE_LtoANK_L = KNE_LtoANK_L.assign(
            CameraZ = df['CameraZ.14'] - df['CameraZ.13']
        )
        KNE_LtoANK_L = KNE_LtoANK_L.assign(
            Norm = np.sqrt(KNE_LtoANK_L['CameraX'] **2
                        + KNE_LtoANK_L['CameraY'] **2
                        + KNE_LtoANK_L['CameraZ'] **2)
        )
        KNE_LtoANK_L['CameraX'] = KNE_LtoANK_L['CameraX'] / KNE_LtoANK_L['Norm']
        KNE_LtoANK_L['CameraY'] = KNE_LtoANK_L['CameraY'] / KNE_LtoANK_L['Norm']
        KNE_LtoANK_L['CameraZ'] = KNE_LtoANK_L['CameraZ'] / KNE_LtoANK_L['Norm']
        #print(KNE_LtoANK_L)

        #ANKLE_LEFT -> FOOT_LEFT
        ANK_LtoFOO_L = pd.DataFrame()
        ANK_LtoFOO_L = ANK_LtoFOO_L.assign(
            CameraX = df['CameraX.15'] - df['CameraX.14']
        )
        ANK_LtoFOO_L = ANK_LtoFOO_L.assign(
            CameraY = df['CameraY.15'] - df['CameraY.14']
        )
        ANK_LtoFOO_L = ANK_LtoFOO_L.assign(
            CameraZ = df['CameraZ.15'] - df['CameraZ.14']
        )
        ANK_LtoFOO_L = ANK_LtoFOO_L.assign(
            Norm = np.sqrt(ANK_LtoFOO_L['CameraX'] **2
                        + ANK_LtoFOO_L['CameraY'] **2
                        + ANK_LtoFOO_L['CameraZ'] **2)
        )
        ANK_LtoFOO_L = ANK_LtoFOO_L.assign(
            Norm = np.sqrt(ANK_LtoFOO_L['CameraX'] **2
                        + ANK_LtoFOO_L['CameraY'] **2
                        + ANK_LtoFOO_L['CameraZ'] **2)
        )
        ANK_LtoFOO_L['CameraX'] = ANK_LtoFOO_L['CameraX'] / ANK_LtoFOO_L['Norm']
        ANK_LtoFOO_L['CameraY'] = ANK_LtoFOO_L['CameraY'] / ANK_LtoFOO_L['Norm']
        ANK_LtoFOO_L['CameraZ'] = ANK_LtoFOO_L['CameraZ'] / ANK_LtoFOO_L['Norm']
        #print(ANK_LtoFOO_L)

        ###新しいCSVファイルを作る
        newCSV = pd.DataFrame()
        #print(newCSV)

        #変換した時間を入れる
        newCSV = newCSV.assign(
            time = df[df.columns[0]]
        )

        #一行目につける値
        header = pd.DataFrame(columns = ['time', 'SPINE_SHOULDER->HEAD', '', '', '',
                          'SPINE_SHOULDER->SHOULDER_RIGHT', '', '', '',
                          'SPINE_SHOULDER->SHOULDER_LEFT', '', '', '',
                          'SPINE_SHOULDER->SPINE_MID', '', '', '',
                          'SHOULDER_RIGHT->ELBOW_RIGHT', '', '', '',
                          'ELBOW_RIGHT->HAND_RIGHT', '', '', '',
                          'SHOULDER_LEFT->ELBOW_LEFT', '', '', '',
                          'ELBOW_LEFT->HAND_LEFT', '', '', '',
                          'SPINE_MID->SPINE_BASE', '', '', '',
                          'SPINE_BASE->HIP_RIGHT', '', '', '',
                          'SPINE_BASE->HIP_LEFT', '', '', '',
                          'HIP_RIGHT->KNEE_RIGHT', '', '', '',
                          'KNEE_RIGHT->ANKLE_RIGHT', '', '', '',
                          'ANKLE_RIGHT->FOOT_RIGHT', '', '', '',
                          'HIP_LEFT->KNEE_LEFT', '', '', '',
                          'KNEE_LEFT->ANKLE_LEFT', '', '', '',
                          'ANKLE_LEFT->FOOT_LEFT', '', '', ''])
        #print(header)

        #Dataframeの結合（axis=1で横に追加 0は縦）
        newCSV = pd.concat([newCSV, SP_SHOtoHEAD, SP_SHOtoSHO_R, SP_SHOtoSHO_L, SP_SHOtoSP_MID,
                            SHO_RtoELB_R, ELB_RtoHAN_R, SHO_LtoELB_L, ELB_LtoHAN_L,
                            SP_MIDtoSP_BASE, SP_BASEtoHIP_R, SP_BASEtoHIP_L,
                            HIP_RtoKNE_R, KNE_RtoANK_R, ANK_RtoFOO_R, HIP_LtoKNE_L,
                            KNE_LtoANK_L, ANK_LtoFOO_L ], axis=1)
        #欠損している行を削除する
        newCSV = newCSV.dropna(how='any')

        #print(newCSV)


        ###書き出し
        vecDir = f + "\\vectorized"
        vecBody = vecDir + "\\body.csv"
        vecLines = vecDir + "\\lines.txt"
        #フォルダを作成
        os.makedirs(vecDir, exist_ok=True)
        #一番上の行
        header.to_csv(vecBody, sep=",",index=False, header=True)
        #それ以降のデータ
        newCSV.to_csv(vecBody, sep=",",index=False, header=True, mode='a')
        #lines.txtのコピー
        #shutil.copy2(linesfile, vecLines)

        ###lines.txtを読んで切る出すプログラムを呼び出し
        #split = ls.LineSplitter(vecBody,vecLines)
        #split.run_alt()
