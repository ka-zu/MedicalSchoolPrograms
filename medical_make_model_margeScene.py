#医大納品用
#学習モデルを作成するプログラム
#傷病者発見のシーンを分けずに学習（シーン1,2をがったい）
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import pandas as pd#csv読み込み
import glob#ファイル検索用

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

#学習・テストデータの作成
'''識別の種類 　-1をつけるとrelu層の影響で0になるから0を何でもないにする
0 : 何でもない   1 : find , safety
2 : awareness   3 : support 5 : check
4 : push        5 : breath
'''

#ここに結合していく
dataDF = pd.DataFrame()
targetDF = pd.DataFrame()

#body.csvのパスを並べる
csvfiles = glob.glob("./data/*/vectorized/body.csv")
linefiles = glob.glob("./data/*/vectorized/lines.txt")

#存在するファイル分ループ
for (f,l) in zip(csvfiles,linefiles):
    #print(f)
    tmpDataDF = pd.read_csv(f, header=1, dtype = 'float32')#読み込み
    tmpDataDF = tmpDataDF.rename(columns={tmpDataDF.columns[0]:"time"})#最初の列名をtimeにする
    tmpLinesDF = pd.read_csv(l, header=None)#読み込み
    #print(tmpLinesDF)
    #識別する値を格納するカラムを追加する
    tmpDataDF = pd.concat([tmpDataDF,pd.DataFrame(columns=['target'])],axis=1)
    #0で初期化
    tmpDataDF['target'] = 0

    ###学習データとそれに対する識別値を与える
    #linesの中身を一行ずつループ
    for index, row in tmpLinesDF.iterrows():

        #識別する値　始まりのフレーム番号　終わりのフレーム番号
        classiStr, start, end = row[0], row[1]-2, row[2]-2
        if(start < 0 or end < 0):
            continue
        if classiStr == "find":
            tmpDataDF.loc[start:end, 'target'] = 1
        elif classiStr == "safety":
            tmpDataDF.loc[start:end, 'target'] = 1
        elif classiStr =="awareness":
            tmpDataDF.loc[start:end, 'target'] = 2
        elif classiStr == "support":
            tmpDataDF.loc[start:end, 'target'] = 3
        elif classiStr =="check":
            tmpDataDF.loc[start:end, 'target'] = 4
        elif ('push' in classiStr):
            tmpDataDF.loc[start:end, 'target'] = 5
        elif ('breath' in classiStr):
            tmpDataDF.loc[start:end, 'target'] = 6

    #欠損している行を削除する
    tmpDataDF = tmpDataDF.dropna(how='any')
    #targetを設定したデータフレームを結合
    dataDF = pd.concat([dataDF, tmpDataDF])

#print(dataDF.iloc[:,0:69])
print(dataDF.shape)

#学習するパラメータデータ準備
batchsize = 1500
datasize = (len(dataDF))
N = 1000


#データと答えを学習する部分とテストする部分に分ける
#データフレーム.valuesで一行ごとの行列になる timeの部分（1列目は入れない）
x_train = dataDF.iloc[:,1:69].values
y_train = dataDF['target'].values

#学習の準備
#MLPクラスで定義した層を使って分類
model = L.Classifier(MLP())
#Adamという最適化法を用いる
optimizer = optimizers.Adam()
#モデルに適用
optimizer.setup(model)

#20個ずつ（エポックサイズを30）で学習・テストする
for epoch in range(100):
    print('epoch % d' % epoch)

    #学習部
    indexes = np.random.permutation(datasize)
    sum_loss, sum_accuracy = 0, 0
    for i in range(0, datasize, batchsize):
        x = Variable(np.asarray(x_train[i : i + batchsize]))
        t = Variable(np.asarray(y_train[i : i + batchsize]))
        optimizer.update(model, x, t)#モデルに対して学習する値とその解を最適化させる
        sum_loss += float(model.loss.data) * batchsize
        sum_accuracy += float(model.accuracy.data) * batchsize
    print('train mean loss={}, accuracy={}'.format(sum_loss / datasize, sum_accuracy / datasize))

#modelをmyMlPの名前で保存
serializers.save_hdf5('myMLP_mergeScene.model',model)
