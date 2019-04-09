import numpy as np
import pandas as pd
import csv

class LineSplitter:

    def __init__(self, csv_fname, splitter_fname):
        self.csv_fname = csv_fname
        self.splitter_fname = splitter_fname
        print(self.csv_fname)
        print(self.splitter_fname)
        with open(csv_fname, 'r') as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            self.header, self.body = np.array(data[:2]), np.array(data[2:])


    def run(self):
        with open(self.splitter_fname, 'r') as f:
            reader = csv.reader(f)
            self.splitters = {row[0]: (int(row[1]), int(row[2])) for row in reader}
        for fname, (beg, end) in self.splitters.items():
            col_num = self.body.shape[0]
            beg = min(max(beg - 3, 0), col_num)
            end = min(max(end - 2, beg), col_num)
            out_fname = ("__%s." % fname).join(self.csv_fname.rsplit('.', 1))
            with open(out_fname, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(self.header)
                writer.writerows(self.body[beg:end])

    #runで動かない時の代わりの関数pandas使う
    def run_alt(self):
        lineDF = pd.read_csv(self.splitter_fname, header=None)#line.txt読み込み
        print(lineDF)
        lineArr = lineDF.values#配列の形にする
        row_num = self.body.shape[0]#行の数を数える
        for (fname, beg, end) in lineArr:#一列ずつ読み込み
            print(fname),print(beg),print(end)
            #はじめ・終わりの数字が配列からはみ出ないように調整
            beg = min(max(beg - 3, 0), row_num)
            end = min(max(end - 2, beg), row_num)
            #出力するファイル名に分類名を書く
            out_fname = ("__%s." % fname).join(self.csv_fname.rsplit('.', 1))
            with open(out_fname, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(self.header)
                writer.writerows(self.body[beg:end])
