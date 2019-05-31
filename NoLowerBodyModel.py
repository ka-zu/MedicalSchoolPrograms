#chainer学習時に下半身を用いない学習を行うモデル
from chainer import Chain
import chainer.links as L
import chainer.functions as F


class NoLowerBodyModel(Chain):
    def __init__(self):#層の様子を定義
        #層：インプット,隠れ層1,隠れ層2,アウトプット
        super(NoLowerBodyModel, self).__init__(#層の移り変わりを示している
            l1=L.Linear(44, 35),#入力は44次元
            l2=L.Linear(35, 35),#中間層は(入力+出力)*(2/3)次元がいいらしい
            l3=L.Linear(35, 8),#出力は8種類で8次元
        )

    def __call__(self, x):#伝番の様子を定義
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
