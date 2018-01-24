# -*- coding: utf-8 -*-
# import  numpy as np
# import  data_preprocessing
# from training_params import *
# import webrtcvad
# import wave
# import vad
# types = {1: np.int8, 2: np.int16, 4: np.int32}
# wav = wave.open('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav', mode="r")
# vad.read_wave('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav')
# (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
# content = wav.readframes(nframes) # 读取全部的帧
# samples = np.fromstring(content, dtype=types[sampwidth]) #将声音文件数据转换为数组矩阵形式
# rtcvad = webrtcvad.Vad()
# rtcvad.is_speech(samples,16000)
import matplotlib.pyplot as plt
import numpy as np
import training_models
import data_preprocessing
import data_extraction
import data_reading
from training_params import *
from sklearn.model_selection import StratifiedKFold
import os
from pylab import*
import scipy
import wave
import pyaudio
import numpy
import pylab

#打开WAV文档，文件路径根据需要做修改
wf = wave.open('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav', mode="r")
#创建PyAudio对象
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
channels=wf.getnchannels(),
rate=wf.getframerate(),
output=True)
nframes = wf.getnframes()
framerate = wf.getframerate()
#读取完整的帧数据到str_data中，这是一个string类型的数据
str_data = wf.readframes(nframes)
wf.close()
#将波形数据转换为数组
# A new 1-D array initialized from raw binary or text data in a string.
wave_data = numpy.fromstring(str_data, dtype=numpy.short)
#将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
wave_data.shape = -1,2
#将数组转置
wave_data = wave_data.T
#time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
#time = numpy.arange(0,nframes)*(1.0/framerate)
#绘制波形图
#pylab.plot(time, wave_data[0])
#pylab.subplot(212)
#pylab.plot(time, wave_data[1], c="g")
#pylab.xlabel("time (seconds)")
#pylab.show()
#
# 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
N=16000
start=0 #开始采样位置
df = framerate/(N-1) # 分辨率
freq = [df*n for n in range(0,N)] #N个元素
wave_data2=wave_data[0][start:start+N]
c=numpy.fft.fft(wave_data2)*2/N
#常规显示采样频率一半的频谱
d=int(len(c)/2)

pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
pylab.show()



#绘制波形图
iemocap_data1= np.load('iemocap_sentences.npy')
signal=iemocap_data1[0]['signal']


# iemocap_data2 = np.load('iemocap_dialog.npy')
# a=iemocap_data2[0]['signal']
# b=data_reading.stereo2mono(iemocap_data2[0]['signal'])

# x,y=librosa.load('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav')
# a,b=wf.read('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav')
#wav = wave.open('D:\IEMOCAP\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F003.wav', mode="r")
#(nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
#
# (Fs, x)= audioBasicIO.readAudioFile('D:\IEMOCAP\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F003.wav')
# sample = audioBasicIO.stereo2mono(x)
# sample=np.array(sample,dtype='int16')
# print(sample)

# samples=audioBasicIO.stereo2mono(samples)
# print(samples)
# import wave
# import numpy as np
# import matplotlib.pyplot as plt
# import Volume as vp
#
# def findIndex(vol,thres):
#     l = len(vol)
#     ii = 0
#     index = np.zeros(4,dtype=np.int16)
#     for i in range(l-1):
#         if((vol[i]-thres)*(vol[i+1]-thres)<0):
#             index[ii]=i
#             ii = ii+1
#     return index[[0,-1]]
#
# fw = wave.open('sunday.wav','r')
# params = fw.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = fw.readframes(nframes)
# waveData = np.fromstring(strData, dtype=np.int16)
# waveData = waveData*1.0/max(abs(waveData))  # normalization
# fw.close()
#
# frameSize = 256
# overLap = 128
# vol = vp.calVolume(waveData,frameSize,overLap)
# threshold1 = max(vol)*0.10
# threshold2 = min(vol)*10.0
# threshold3 = max(vol)*0.05+min(vol)*5.0
#
# time = np.arange(0,nframes) * (1.0/framerate)
# frame = np.arange(0,len(vol)) * (nframes*1.0/len(vol)/framerate)
# index1 = findIndex(vol,threshold1)*(nframes*1.0/len(vol)/framerate)
# index2 = findIndex(vol,threshold2)*(nframes*1.0/len(vol)/framerate)
# index3 = findIndex(vol,threshold3)*(nframes*1.0/len(vol)/framerate)
# end = nframes * (1.0/framerate)
#
# plt.subplot(211)
# plt.plot(time,waveData,color="black")
# plt.plot([index1,index1],[-1,1],'-r')
# plt.plot([index2,index2],[-1,1],'-g')
# plt.plot([index3,index3],[-1,1],'-b')
# plt.ylabel('Amplitude')
#
# plt.subplot(212)
# plt.plot(frame,vol,color="black")
# plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
# plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
# plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
# plt.legend()
# plt.ylabel('Volume(absSum)')
# plt.xlabel('time(seconds)')
# plt.show()