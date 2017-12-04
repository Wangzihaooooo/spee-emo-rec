import matplotlib
import wave
import numpy as np
import os
import audioBasicIO
from training_params import *

#删除iemocap中不可用的文件
def process_iemocap_data(dir):
    path_collection = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            path_collection.append(fullpath)
    for file in path_collection:
        extension = os.path.splitext(file)[0]
        if "." in extension:
            os.remove(file)

def read_iemocap_data(dir):
  data = []
  sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
  for session in sessions:
    print (session)
    path_to_wav = dir + session + '/dialog/wav/'
    path_to_emotions = dir + session + '/dialog/EmoEvaluation/'
    path_to_transcriptions = dir + session + '/dialog/transcriptions/'

    files = os.listdir(path_to_wav)#返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    files = [os.path.splitext(f)[0] for f in files]#分离文件名与扩展名,默认返回(fname,fextension)元组，可做分片操作
    print (len(files))
    print (files)
    for f in files:
      emotions = get_emotions(path_to_emotions, f + '.txt')
      wav = open_wav(path_to_wav, f + '.wav')
      sample = split_wav(wav, emotions)

      transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
      for ie, e in enumerate(emotions):
        if e['emotion'] in available_emotions:
            e['signal']=[]
            e['signal'].append(sample[ie]['right'])
            e['signal'].append(sample[ie]['left'])
            e['signal']=np.array(e['signal']).T
            e['signal']=audioBasicIO.stereo2mono(e['signal'])
            e['transcription'] = transcriptions[e['id']]
            data.append(e)
  return data

def get_emotions(path_to_emotions, filename):
    f = open(path_to_emotions + filename, 'r').read()
    # 'r'读模式、'w'写模式、'a'追加模式、'b'二进制模式、'+'读/写模式 默认是读模式
    # read( )：表示读取全部内容 readline( )：表示逐行读取
    f = np.array(f.split('\n'))  # np.append(np.array(['']), np.array(f.split('\n')))
    c = 0
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i] + 1:idx_n[i + 1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[
                   head.find(filename[:-4]) + len(filename[:-4]) + 1:head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo})
    return emotion

def open_wav(path_to_wav, filename):
    types = {1: np.int8, 2: np.int16, 4: np.int32}
    wav = wave.open(path_to_wav + filename, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes) # 读取全部的帧
    samples = np.fromstring(content, dtype=types[sampwidth]) #将声音文件数据转换为数组矩阵形式
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples

#获取左右双声道
def split_wav(wav, emotions):
    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav
    duration = nframes / framerate #音频持续时间

    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']
        e['right'] = right[int(start * framerate):int(end * framerate)]
        e['left'] = left[int(start * framerate):int(end * framerate)]
        frames.append({'left': e['left'], 'right': e['right']})
    return frames

#获取音频对应的文本
def get_transcriptions(path_to_transcriptions, filename):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}

    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1 + 2:]
        transcription[ind_id] = ind_ts
    return transcription

#根据key返回data对应的键值
def get_field(data, key):
    return np.array([e[key] for e in data])