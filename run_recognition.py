import matplotlib.pyplot as plt
import numpy as np
import training_models
import data_preprocessing
import data_extraction
import data_reading
import training_params
from sklearn.model_selection import StratifiedKFold
import os
#读取语料库对话音频数据
np.random.seed(100)
X =[]
Y =[]
preds = []  # 预期结果
trues = []  # 真实结果
energies = []
data_type=1 #{'iemocap_sentences':1,'iemocap_dialog':0}
get_features=0 #是否重新提取特征

if(data_type):
    npy_name='iemocap_sentences.npy'
    if(os.path.exists(npy_name)):
        if get_features:
            iemocap_data=np.load(npy_name)
            data_extraction.get_features(iemocap_data, training_params.path_to_sentences_samples)
        X, Y = data_extraction.get_sample(training_params.path_to_sentences_samples)
    else:
        iemocap_data = np.array(data_reading.get_iemocap_sentences(training_params.path_to_iemocap))
        np.save(npy_name, iemocap_data)
else:
    npy_name = 'iemocap_dialog.npy'
    if (os.path.exists(npy_name)):
        if get_features:
            iemocap_data = np.load(npy_name)
            data_extraction.get_features(iemocap_data, training_params.path_to_dialog_samples)
        X, Y = data_extraction.get_sample(training_params.path_to_dialog_samples)
    else:
        iemocap_data = np.array(data_reading.get_iemocap_dialog(training_params.path_to_iemocap))
        np.save(npy_name, iemocap_data)


'''5-fold  cross_validation（K-折交叉验证）'''
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
for train, test in kfold.split(X, Y):
    train_x=X[train]
    train_y=Y[train]
    test_x=X[test]
    test_y=Y[test]
    train_x=data_preprocessing.normalize(train_x)
    test_x=data_preprocessing.normalize(test_x)

    timestep = 50
    train_x = data_preprocessing.pad_sequence(train_x, timestep)
    test_x = data_preprocessing.pad_sequence(test_x, timestep)
    train_y_cat = data_preprocessing.to_categorical(train_y)
    test_y_cat = data_preprocessing.to_categorical(test_y)

    model = training_models.train_lstm(train_x, train_y_cat, test_x, test_y_cat)
    scores = model.predict(test_x)
    prediction = np.array([training_params.available_emotions[np.argmax(t)] for t in scores])  # np.argmax(t) 返回最值所在的索引
    #print(prediction[prediction ==test_y].shape[0] / float(prediction.shape[0]))
    for i in range(len(prediction)):
        preds.append(prediction[i])
        trues.append(test_y[i])

preds = np.array(preds) # 预期结果
trues = np.array(trues) # 真实结果
print ('Total accuracy: ', preds[preds == trues].shape[0] / float(preds.shape[0]))

#音调由声波的频率决定，频率越高音调越高。响度由声波的振幅决定，振幅越高响度越大。音色是由波形的“形”决定的。
#绘制训练结果的热图 plots confusion matrix aomparing prediction and expected output

#绘制波形图
def get_oscillogram(signal):
    time = np.arange(0, len(signal)) * (1.0 /training_params.framerate )
    plt.plot(time, signal,c='green')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude(db)")
    plt.grid('on')  # 标尺，on：有，off:无。
    plt.show()

#绘制动态 声谱图(横坐标代表时间，纵坐标代表频率，颜色代表振幅)
def get_spectrogram(signal):
    plt.specgram(signal, Fs=training_params.framerate, scale_by_freq=True, sides='default')
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.show()

#绘制静态 频谱图(横坐标代表频率，纵坐标代表振幅)
def get_spectrum(signal):
    plt.subplots_adjust(signal, Fs=training_params.framerate, scale_by_freq=True, sides='default')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.grid('on')
    plt.show()

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
get_heat_map(training_params.available_emotions,preds,trues)

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
