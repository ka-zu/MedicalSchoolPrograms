#医大納品用
#入力されたデータに対して結果を出力するプログラム
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import pandas as pd
import glob#ファイル検索用
from statistics import mode#結果修正の最頻値を出す用

#ニューラルネットの定義
class MLP(Chain):
    def __init__(self):#層の様子を定義
        #層：インプット,隠れ層1,隠れ層2,アウトプット
        super(MLP, self).__init__(
            l1=L.Linear(68, 200),#68個のベクトルで69次元
            l2=L.Linear(200, 200),#層の移り変わりを示している
            l3=L.Linear(200, 7),#出力は7種類で7次元
        )

    def __call__(self, x):#伝番の様子を定義
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

#学習したモデルとデータですぐにアウトプットできる関数
def predict(model, x_data):
        x = Variable(x_data.astype(np.float32))
        y = model.predictor(x)
        return np.argmax(y.data, axis=1)

#最頻値フィルタ関数
#ある点の後5つの値を見て最頻値を値とする
def mode_filter(array):
    tmpArray = []
    hist = 0#前回の値
    for i in range(0,len(array)):
        if(i < len(array)-5):
            try:
                tmpArray.append(mode(array[i:i+5]))
                hist = mode(array[i:i+5])
            except:
                #最頻値が求まらなかったら
                tmpArray.append(hist)
        else:
            tmpArray.append(array[i])

    return tmpArray

#最頻値フィルタ関数
#範囲を任意の定数にした
def mode_filter(array,mode_range):
    tmpArray = []
    hist = 0#前回の値
    for i in range(0,len(array)):
        if(i < len(array)-mode_range):
            try:
                tmpArray.append(mode(array[i:i+mode_range]))
                hist = mode(array[i:i+mode_range])
            except:
                #最頻値が求まらなかったら
                tmpArray.append(hist)
        else:
            tmpArray.append(array[i])

    return tmpArray

#平滑化（移動平均フィルタ）
def smoothing_filter(data_frame):
    #前後1つと平均をとる
    tmpDataFrame = data_frame.rolling(window=3, center=True).mean()
    #NaNとなった部分を前の値から持ってくる
    tmpDataFrame.iloc[0] = data_frame.iloc[0]
    tmpDataFrame.iloc[len(tmpDataFrame)-1] = data_frame.iloc[len(tmpDataFrame)-1]
    #ずれたTimeデータを持ってくる
    tmpDataFrame.iloc[:,0] = data_frame.iloc[:,0]

    return tmpDataFrame

#テストしたいファイルをtestフォルダの中に
testCSV = glob.glob("./test/*/vectorized/body.csv")

#print(testCSV)

print("[start]")

#読み込んだファイル分ループ
for f in testCSV:

    #ファイル名の取得
    filename = f.split('\\')#フォルダ区切り記号で分割
    filename = filename[1]#二番目が読み込んでいるファイル名
    #print(filename)

    testDataDF = pd.read_csv(f, header=1, dtype='float32')#CSVデータを読み込み
    testDataDF = testDataDF.rename(columns={testDataDF.columns[0]:"time"})#最初の列名をtimeにする

    #識別する値を格納するカラムを追加する
    testDataDF = pd.concat([testDataDF, pd.DataFrame(columns=['target'])], axis=1)

    #識別したいデータを平滑化する（移動平均フィルタ）
    testDataDF = smoothing_filter(testDataDF)
    testDataDF = smoothing_filter(testDataDF)
    testDataDF = smoothing_filter(testDataDF)

    #識別したいデータ
    testData = testDataDF.iloc[:,1:69].values
    predictData = np.empty((0,68), float)
    for i in range(0,len(testData)):
        predictData = np.append(predictData, np.array([testData[i]]), axis=0)



    #作ったモデルをロードして結果を分類する
    #型を作ってから（学習時にchainer.linksのClassifierを使った場合同じ関数で作る）
    model = L.Classifier(MLP())
    #ロード
    serializers.load_hdf5('myMLP_mergeScene.model',model)

    #識別
    result = predict(model, testData)

    result = np.array(result)


    ###データの補修部
    #後ろの5データをみて一番多い値を識別番号にする
    #一番多い値が出せない場合、前回入れた値と同じにする
    #amendment:修正
    amendResult = []
    hist = 0#前回の値
    for i in range(0,len(result)):
        if(i < len(result)-5):
            try:
                amendResult.append(mode(result[i:i+5]))
                hist = mode(result[i:i+5])
            except:
                #最頻値が求まらなかったら
                amendResult.append(hist)
        else:
            amendResult.append(result[i])

    #2回目の最頻値
    amendResult2 = mode_filter(amendResult,50)
    #3回目の最頻値フィルタ
    amendResult3 = mode_filter(amendResult2,50)

    ###結果の出力
    #出力用のデータフレーム
    resultDF = pd.DataFrame()
    #配列をSeriesに変換
    resultSR = pd.Series(result, name="result")
    amendSR = pd.Series(amendResult, name="amendResult")
    amendSR2 = pd.Series(amendResult2, name="amendResult2")
    amendSR3 = pd.Series(amendResult3, name="amendResult3")
    #時間と答えと識別番号をくっつける
    resultDF = pd.concat([resultDF, testDataDF['time'], resultSR, amendSR, amendSR2, amendSR3], axis=1)
    #csv出力
    resultDF.to_csv("./result//"+filename+"_result_mergeScene.csv", sep=",", index=False, header=True, mode='w')

    print("-file : "+ filename +" end")

print("[end]")
