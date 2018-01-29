import librosa
import matplotlib.pyplot as plt
import numpy as np
import training_models
import data_preprocessing
import data_extraction
import data_reading
from training_params import *
from sklearn.model_selection import StratifiedKFold
import os
import training_params
import wave
import scipy.io.wavfile as wavfile
import shutil


# filename='C:/Users/wangzi/Desktop/大创资料/语料库/chinese_dataset/liuchanhg/fear/210.wav'
# y, sr = librosa.load(filename,sr=16000)
# wf = wave.open(filename, mode="r")
# nframes = wf.getnframes()
# framerate = wf.getframerate()
# #读取完整的帧数据到str_data中，这是一个string类型的数据
# str_data = wf.readframes(nframes)
# wave_data = np.fromstring(str_data, dtype=np.short)
# fs, signal = wavfile.read(filename)
# # Signal normalization
# signal = np.double(signal)
# signal = signal / (2.0 ** 15) #2的15次方
# DC = signal.mean()  #求平均值
# STD = np.std(signal) #求标准差
# MAX = (np.abs(signal)).max()
# signal = (signal - DC) / STD
# a=1
