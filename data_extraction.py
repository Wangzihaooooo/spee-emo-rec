import numpy as np
import csv
from training_params import *
import data_features
import data_preprocessing
import os
import training_params
import shutil
#提取音频特征 并存放到csv文件中
def extract_features(data, save_path,save=True):
    print('begin extract.....')
    for index, item in enumerate(data):
        if index % 200 == 0:
            print(index, ' out of ', len(data))

        window_sec = 0.2
        window_n = int(framerate * window_sec)
        samples = np.array(item['signal'], dtype=data.dtype)
        st_features = data_features.stFeatureExtraction(samples, framerate, window_n, window_n / 2).T

        x = []
        y = []
        # 在每行后面添加情绪标签
        for f in st_features:
            if f[1] > 1.e-4:
                x.append(f)
                y.append(item['emotion'])
        x = np.array(x, dtype=float)
        y = np.array(y)

        if save:
            try:
                shutil.rmtree(save_path)
            except:
                os.mkdir(save_path)
            save_sample(x, y, save_path + item['id'] + '.csv')
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
        if (len(row) != 0) and (row[-1] in training_params.available_emotions):#因为csv文件中存在一些行为空值的情况
            x.append(row[:-1])
            y.append(row[-1])
  return np.array(x, dtype=float), np.array(y)

#对读取得到的数据进行处理
def get_sample(path):
  tx = []
  ty = []
  csv_file_list=os.listdir(path)
  for csv_file in csv_file_list:
      path_to_csvfile=path+csv_file
      if (os.path.exists(path_to_csvfile)):
          x, y = load(path_to_csvfile)
          if  len(x) > 0 and len(y) > 0:
              tx.append(np.concatenate((x[:,2:4],x[:,8:]),axis=1))
              ty.append(y[0])

  tx = np.array(tx)
  ty = np.array(ty)
  return tx, ty

