#BLS決定結果の数字列から切り出し点を作成する
import numpy as np
import pandas as pd
import glob
from statistics import mode#結果修正の最頻値を出す用
import csv

#ある点の前後の座標の差の平均(前後の微分の平均)
def diffeAve(arr):
    #print("start diffe")
    #print(arr)
    #微分値を格納する配列
    deffeArr = []
    #前回の切り出したいタイミングを保存
    splitFrame = -30
    #切り出しポイントを保存
    pointArr = []


    #最初の一つ
    deffeArr.append(0)

    for i in range(1,len(arr)-1):
        front = abs(arr[i] - arr[i-1])#前の微分
        rear = abs(arr[i+1] - arr[i])#後ろの微分
        #前回の切り出し点より30フレーム離れていて、微分平均が0.5より大きい
        if(abs(i-splitFrame) > 30 and (front+rear)/2 >= 0.5):
            deffeArr.append((front+rear)/2)
            pointArr.append(i)
            splitFrame = i
        else:
            deffeArr.append(0)

    deffeArr.append(0)
    return pointArr

def main():
    #読み込むCSV
    resultCSVs = glob.glob("result/*.csv")
    #保存先のフォルダ名
    dataFiles = [s.split("_",)[0] for s in resultCSVs]
    dataFiles = [s.split("\\",)[1] for s in dataFiles]

    print("[start]")

    #ファイル数分ループ
    for i in range(len(resultCSVs)):
        #微分した結果を格納する配列
        deffeResult = []
        pointResult = []
        #結果を出力するときの配列
        splitResult = []
        #正しいシーンの順番かどうかを調べるフラグ　0:正しい　１：正しくない
        correctFlg = 1

        #シーン番号設定用
        #結果の数字列読み込み
        resultDF = pd.read_csv(resultCSVs[i])
        bodyDF = pd.read_csv("./test/"+dataFiles[i]+"/vectorized/body.csv", header=1, dtype='float32')
        bodyDF = bodyDF.rename(columns={bodyDF.columns[0]:"time"})

        #コピー
        amendResultArr = resultDF["amendResult"]
        pointResult = diffeAve(amendResultArr)

        #切り出し点のフォーマットを作成
        No = 1#行番号
        start = 0#各シーン始まりののフレーム番号
        splitStart = 0#切り出しの最初のフレーム番号
        end = pointResult[0]#各終わりのフレーム番号
        sceneNum = 0
        sceneNumHist = 0#前回のシーン番号を保存
        sceneNumDiff = 0#シーン番号の差分を保存
        for j in range(len(pointResult)):
            end = pointResult[j]

        #modeでexceptionが起きた時の処理用
        tmpEnd = end
            #シーン番号決定
            while(True):
                try:
                    sceneNum = mode(amendResultArr[start:tmpEnd])
                    return
                except:
                    print("最頻値がかぶりました")
                    tmpEnd -= 1

            sceneNumDiff = sceneNum - sceneNumHist
            #差が0(同じシーン)または1(次のシーン)でない場合(シーンが飛んだ場合)
            if((sceneNumDiff != 0) or (sceneNumDiff != 1)):
                if(sceneNum == 0):#何でもないシーンではない場合
                    correctFlg = 1;#正しくないと判断

            if(sceneNumDiff == 0):#前のシーンと同じだったら
                start = end+1
                continue

            #判別結果をフォーマットに合うように保存
            if(sceneNum != 0):#シーン0はいらない
                splitResult.append((No,bodyDF["time"][start],bodyDF["time"][end],start,end,sceneNum))
            splitresultDF = pd.DataFrame(splitResult)
            splitresultDF.columns = ["No","TimeStart","TimeEnd","FrameStart","FrameEnd","SceneNum"]

            No += 1
            start = end+1#次の始まりのフレーム
            splitStart = end+111
            sceneNumHist = sceneNum

        #切り出し点の出力
        with open("./test/"+dataFiles[i]+"/scene.csv","w") as f:
            writer = csv.writer(f, lineterminator="\n")
            header = ["No.","Time"," ","Frame"," "," "]
            header2 = [correctFlg,"start","end","start","end","AI Judge"]
            writer.writerow(header)
            writer.writerow(header2)
            #writer.writerows(splitResult)
            #writer.writerows(fixSplit)

        splitresultDF.to_csv("./test/"+dataFiles[i]+"/scene.csv",mode="a",header=False,index=False)

        print("-file : "+dataFiles[i]+" end")

#実行
main()
print("[end]")
