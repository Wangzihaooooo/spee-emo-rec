import matplotlib.pyplot as plt
import wave
import numpy as np
import training_models
import data_reading
import data_preprocessing
from training_params import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale

#读取语料库对话音频数据
seed=100
np.random.seed(seed)
data = np.array(data_reading.read_iemocap_data(path_to_samples))
ids = data_reading.get_field(data, 'id')
data_preprocessing.get_features(data,path_to_samples)
np.save("data_ids.npy",ids)

#ids=np.load("data_ids.npy")
X,Y=data_preprocessing.get_sample(ids[:], path_to_samples)
preds = []  # 预期结果
trues = []  # 真实结果
energies = []
'''5-fold  cross_validation（K-折交叉验证）'''
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, Y):
    train_x=X[train]
    train_y=Y[train]
    test_x=X[test]
    test_y=Y[test]
    train_x=data_preprocessing.normalize(train_x)
    test_x=data_preprocessing.normalize(test_x)

    timestep = 32
    train_x = data_preprocessing.pad_sequence(train_x, timestep)
    test_x = data_preprocessing.pad_sequence(test_x, timestep)
    train_y_cat = data_preprocessing.to_categorical(train_y)
    test_y_cat = data_preprocessing.to_categorical(test_y)

    model = training_models.train_lstm(train_x, train_y_cat, test_x, test_y_cat)
    scores = model.predict(test_x)
    prediction = np.array([available_emotions[np.argmax(t)] for t in scores])  # np.argmax(t) 返回最值所在的索引
    print(prediction[prediction ==test_y].shape[0] / float(prediction.shape[0]))
    for i in range(len(prediction)):
        preds.append(prediction[i])
        trues.append(test_y[i])

preds = np.array(preds) # 预期结果
trues = np.array(trues) # 真实结果
print ('Total accuracy: ', preds[preds == trues].shape[0] / float(preds.shape[0]))

#绘制训练结果的热图 plots confusion matrix aomparing prediction and expected output
def get_heat_map(available_emotions,preds,trues):
    class_to_class_precs = np.zeros((len(available_emotions), len(available_emotions)), dtype=float)
    for cpi, cp in enumerate(available_emotions):
        for cti, ct in enumerate(available_emotions):
            if trues[trues == ct].shape[0] > 0:
                class_to_class_precs[cti, cpi] = preds[(preds == cp) * (trues == ct)].shape[0] / float(
                    trues[trues == ct].shape[0])
            else:
                class_to_class_precs[cti, cpi] = 0.

    fig, ax = plt.subplots()  # 函数返回一个figure图像和一个子图ax的array列表
    ax.pcolor(class_to_class_precs.T, cmap=plt.cm.Blues)  # 设置颜色
    # 设置横纵坐标的数值
    ax.set_xticklabels(available_emotions, minor=False)
    ax.set_yticklabels(available_emotions, minor=False)
    # put the major ticks at the middle of each cell 设置横纵坐标大小
    ax.set_xticks(np.arange(len(available_emotions)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(available_emotions)) + 0.5, minor=False)
    # 设置每个方格的数值
    for i in range(len(available_emotions)):
        for j in range(len(available_emotions)):
            plt.text(i + 0.4, j + 0.4, str(class_to_class_precs[i, j])[:5])

    plt.savefig('training_result.png')
    plt.close()
get_heat_map(available_emotions,preds,trues)

#统计不同情绪的音频数量
# #emotions = get_field(data, 'emotion')
# print (np.unique(emotions))
# for i in range(len(available_emotions)):
#   print (available_emotions[i], emotions[emotions == available_emotions[i]].shape[0])

# energies = []
# for i in xrange(train_x.shape[0]):
#   energies.append(np.mean(train_x[i][:, 1]))

# quantile = 0.5
# largest_energy_idx = np.argsort(energies)[:int(quantile*len(energies))]

# train_x = train_x[largest_energy_idx]
# train_y = train_y[largest_energy_idx]


# train_x = normalize(train_x)
# test_x = normalize(test_x)

# train_x, train_y = grow_sample(train_x, train_y, 10000)
# train_x, train_y = reshape_for_dense(train_x, train_y)

# lengths = {}
# for i in xrange(train_x.shape[0]):
#   if train_x[i].shape[0] in lengths.keys():
#     lengths[train_x[i].shape[0]] += 1
#   else:
#     lengths[train_x[i].shape[0]] = 1

# for k, v in lengths.items():
#   print k, v
