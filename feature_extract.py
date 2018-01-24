#-*- coding:utf-8 -*-
import librosa
import numpy as np
import scipy.io.wavfile as wf
import pandas as pd
from collections import Counter
import os
import training_params

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size //2)

def extract_features_cnn(bands=60, frames=41):

    log_spectograms_ang = []
    log_spectograms_exc = []
    log_spectograms_neu = []
    log_spectograms_sad = []

    window_size = 512 * (frames - 1)

    class_data = pd.read_csv('samples.csv')
    classes = class_data.loc[:, ['sample','class']].as_matrix()
    classes_dict = dict(classes)

    str1 = 'D:\\file\\文件备份\\IEMOCAP_full_release.tar\\IEMOCAP_full_release\\IEMOCAP_full_release\\Session'
    str2 =  '\sentences\wav'

    for i in range(1, 6):
        dir1 = str1 + str(i) + str2
        dirs = os.listdir(dir1)
        for j in dirs:
            if j.startswith('Ses'):
                dir2 = dir1 + '\\' + j
                filenames = os.listdir(dir2)
                for file in filenames:
                    if file.startswith('Ses') & file.endswith('.wav'):
                        wav = dir2 + '\\' + file
                        lable = classes_dict[file[:-4]]
                        if (lable == 'ang') | (lable == 'exc') | (lable == 'neu') | (lable == 'sad'):
                            print("extracting feature from " + wav)
                            audio_clip, sr = librosa.load(wav)
                            for (start, end) in windows(audio_clip, window_size):
                                if (len(audio_clip[start:end]) == window_size):
                                    audio_signal = audio_clip[start:end]
                                    mel_spec = librosa.feature.melspectrogram(audio_signal, n_mels=bands)
                                    log_spec = librosa.logamplitude(mel_spec)
                                    log_spec = log_spec.T.flatten()[:, np.newaxis].T

                                    if lable == 'ang':
                                        log_spectograms_ang.append(log_spec)
                                    elif lable == 'exc':
                                        log_spectograms_exc.append(log_spec)
                                    elif lable == 'neu':
                                        log_spectograms_neu.append(log_spec)
                                    elif lable == 'sad':
                                        log_spectograms_sad.append(log_spec)

    log_spectograms_ang = np.asarray(log_spectograms_ang).reshape(len(log_spectograms_ang), bands, frames, 1)
    log_spectograms_exc = np.asarray(log_spectograms_exc).reshape(len(log_spectograms_exc), bands, frames, 1)
    log_spectograms_neu = np.asarray(log_spectograms_neu).reshape(len(log_spectograms_neu), bands, frames, 1)
    log_spectograms_sad = np.asarray(log_spectograms_sad).reshape(len(log_spectograms_sad), bands, frames, 1)

    features_cnn_ang = np.concatenate((log_spectograms_ang, np.zeros(np.shape(log_spectograms_ang))), axis=3)
    features_cnn_exc = np.concatenate((log_spectograms_exc, np.zeros(np.shape(log_spectograms_exc))), axis=3)
    features_cnn_neu = np.concatenate((log_spectograms_neu, np.zeros(np.shape(log_spectograms_neu))), axis=3)
    features_cnn_sad = np.concatenate((log_spectograms_sad, np.zeros(np.shape(log_spectograms_sad))), axis=3)

    for i in range(len(features_cnn_ang)):
        features_cnn_ang[i, :, :, 1] = librosa.feature.delta(features_cnn_ang[i, :, :, 0])
    for i in range(len(features_cnn_exc)):
        features_cnn_exc[i, :, :, 1] = librosa.feature.delta(features_cnn_exc[i, :, :, 0])
    for i in range(len(features_cnn_neu)):
        features_cnn_neu[i, :, :, 1] = librosa.feature.delta(features_cnn_neu[i, :, :, 0])
    for i in range(len(features_cnn_sad)):
        features_cnn_sad[i, :, :, 1] = librosa.feature.delta(features_cnn_sad[i, :, :, 0])

    np.save('features_cnn_ang', features_cnn_ang)
    np.save('features_cnn_exc', features_cnn_exc)
    np.save('features_cnn_neu', features_cnn_neu)
    np.save('features_cnn_sad', features_cnn_sad)



def extract_features_mfcc(bands=20, frames=41):

    window_size = 512 * (frames - 1)

    mfcc_ang = []
    mfcc_exc = []
    mfcc_neu = []
    mfcc_sad = []

    class_data = pd.read_csv('samples.csv')
    classes = class_data.loc[:, ['sample','class']].as_matrix()
    classes_dict = dict(classes)

    str1 = 'D:\\file\\文件备份\\IEMOCAP_full_release.tar\\IEMOCAP_full_release\\IEMOCAP_full_release\\Session'
    str2 =  '\sentences\wav'

    for i in range(1, 6):
        dir1 = str1 + str(i) + str2
        dirs = os.listdir(dir1)
        for j in dirs:
            if j.startswith('Ses'):
                dir2 = dir1 + '\\' + j
                filenames = os.listdir(dir2)
                for file in filenames:
                    if file.startswith('Ses') & file.endswith('.wav'):
                        wav = dir2 + '\\' + file
                        lable = classes_dict[file[:-4]]
                        if (lable == 'ang') | (lable == 'exc') | (lable == 'neu') | (lable == 'sad'):
                            print("extracting feature from " + wav)
                            audio_clip, sr = librosa.load(wav)
                            for (start, end) in windows(audio_clip, window_size):
                                if (len(audio_clip[start:end]) == window_size):
                                    signal = audio_clip[start:end]
                                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=bands).T.flatten()[:,
                                           np.newaxis].T
                                    if lable == 'ang':
                                        mfcc_ang.append(mfcc)
                                    elif lable == 'exc':
                                        mfcc_exc.append(mfcc)
                                    elif lable == 'neu':
                                        mfcc_neu.append(mfcc)
                                    elif lable == 'sad':
                                        mfcc_sad.append(mfcc)

    mfcc_ang = np.asarray(mfcc_ang).reshape(len(mfcc_ang), bands, frames)
    mfcc_exc = np.asarray(mfcc_exc).reshape(len(mfcc_exc), bands, frames)
    mfcc_neu = np.asarray(mfcc_neu).reshape(len(mfcc_neu), bands, frames)
    mfcc_sad = np.asarray(mfcc_sad).reshape(len(mfcc_sad), bands, frames)

    np.save('mfcc_ang', mfcc_ang)
    np.save('mfcc_exc', mfcc_exc)
    np.save('mfcc_neu', mfcc_neu)
    np.save('mfcc_sad', mfcc_sad)

# extract_features()

# filename = 'D:\\file\文件备份\IEMOCAP_full_release.tar\IEMOCAP_full_release\I' \
#            'EMOCAP_full_release\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F013.wav'
# y, sr = librosa.load(filename, sr = 16000)
# print(sr)
# print(y)
# rate, data = wf.read(filename)
# print(rate)
# print(data)

#extract_features_cnn(bands=60, frames=41)