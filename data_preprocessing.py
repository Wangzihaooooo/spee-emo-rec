import numpy as np
import csv
from training_params import *
from keras.utils import np_utils
from keras.preprocessing import sequence
import webrtcvad
import collections
import contextlib
import sys
import wave

weighting_type = 'hamming'
def weighting(x):
  if weighting_type == 'cos':
    return x * np.sin(np.pi * np.arange(len(x)) / (len(x) - 1))
  elif weighting_type == 'hamming':
    return x * (0.54 - 0.46 * np. cos(2. * np.pi * np.arange(len(x)) / (len(x) - 1)))
  else:
    return x

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
        if ts >= x[i].shape[0]:
            x0[ts - x[i].shape[0]:, :] = x[i]
        else:
            maxe = np.sum(x[i][0:ts, 0])
            for j in range(x[i].shape[0] - ts):
                if np.sum(x[i][j:j + ts, 0]) > maxe:
                    x0 = x[i][j:j + ts, :]
                    maxe = np.sum(x[i][j:j + ts, 0])
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

def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: example.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    audio, sample_rate = read_wave(args[1])
    vad = webrtcvad.Vad(int(args[0]))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):

        path = 'D:/IEMOCAP/Session1/chunk-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)

def read_wave(path):
    # (Fs, x) = audioBasicIO.readAudioFile('D:\IEMOCAP\Session1\dialog\wav\Ses01F_impro01.wav')
    # sample = audioBasicIO.stereo2mono(x)
    # samples = np.array(sample, dtype='int16')
    # wav = data_reading.open_wav(path  , '.wav')
    #
    # (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav
    # samples=audioBasicIO.stereo2mono(samples)
    return  16000

def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        sys.stdout.write(
            '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

