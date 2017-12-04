import numpy as np
import csv
import calculate_features as cf
from training_params import *
from keras.utils import np_utils
from keras.preprocessing import sequence

#讲标签y进行向量化
def to_categorical(y):
    y_cat = np.zeros((len(y), len(available_emotions)), dtype=int)
    for i in range(len(y)):
        y_cat[i, :] = np.array(np.array(available_emotions) == y[i], dtype=int)
    return y_cat

#进行序列填充
def pad_sequence(x, ts):
    #return  sequence.pad_sequences(x,ts)
    xp = []
    for i in range(x.shape[0]):
        x0 = np.zeros((ts, x[i].shape[1]), dtype=float)
        if ts > x[i].shape[0]:
            x0[ts - x[i].shape[0]:, :] = x[i]
        else:
            maxe = np.sum(x[i][0:ts, 1])
            for j in range(x[i].shape[0] - ts):
                if np.sum(x[i][j:j + ts, 1]) > maxe:
                    x0 = x[i][j:j + ts, :]
                    maxe = np.sum(x[i][j:j + ts, 1])
        xp.append(x0)
    return np.array(xp)

#对数据进行min-max标准化 x=(x-min)/(max-min)  另一种为Z-score标准化方法:均值（mean）和标准差（standard deviation） x=(x-mean)/std
def normalize(x):
  gminx = np.zeros(x[0].shape[1]) + 1.e5
  gmaxx = np.zeros(x[0].shape[1]) - 1.e5
  for i in range(x.shape[0]):
    q = x[i]
    minx = np.min(q, axis=0)
    maxx = np.max(q, axis=0)

    for s in range(x[0].shape[1]):
      if gminx[s] > minx[s]:
        gminx[s] = minx[s]
      if gmaxx[s] < maxx[s]:
        gmaxx[s] = maxx[s]

  for i in range(x.shape[0]):
    for s in range(x[0].shape[1]):
      x[i][:, s] = (x[i][:, s] - gminx[s]) / float(gmaxx[s] - gminx[s])
  return x

#提取音频特征 并存放到csv文件中
def get_features(data, path,save=True):
  failed_samples = []
  for index, item  in enumerate(data):
    if index % 1000 == 0:
      print (index, ' out of ', len(data))
    st_features = cf.calculate_features(item ['signal'], framerate).T
    x = []
    y = []
    for f in st_features:
      if f[1] > 1.e-4:
        x.append(f)
        y.append(item ['emotion'])
    x = np.array(x, dtype=float)
    y = np.array(y)

    if save:
      save_sample(x, y, path + item ['id'] + '.csv')
  return x, y

#讲特征数据存储到csv文件中 并在x的每一行后加上相对应的y
def save_sample(x, y, name):
  with open(name, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for i in range(x.shape[0]):
      row = x[i, :].tolist()
      row.append(y[i])
      w.writerow(row)

#加载csv文件，读入训练数据
def load(name):
  with open(name, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in r:
        if (len(row) != 0):#因为csv文件中存在一些行为空值的情况
            x.append(row[:-1])
            y.append(row[-1])
  return np.array(x, dtype=float), np.array(y)

#对读取得到的数据进行处理
def get_sample(idx, path):
  tx = []
  ty = []
  for i in idx:
    x, y = load(path + '/' + i + '.csv')
    if 0< len(x) < 40 and len(y)>0 :
      tx.append(np.array(x, dtype=float))
      ty.append(y[0])

  tx = np.array(tx)
  ty = np.array(ty)
  return tx, ty

def grow_sample(x, y, n=10000):
  xg = []
  yg = []
  eps = 5.*1.e-2
  for i in range(n):
    j = np.random.randint(x.shape[0])
    x0 = x[j]
    x0 += eps * np.random.normal(0, 1, size=x0.shape)
    y0 = y[j]
    xg.append(x0)
    yg.append(y0)
  return np.array(xg), np.array(yg)

def reshape_for_dense(x, y):
  j = 0
  xr = []
  yr = []
  for i in range(x.shape[0]):
    for k in range(x[i].shape[0]):
      xr.append(x[i][k, :])
      yr.append(y[i])
  return np.array(xr), np.array(yr)