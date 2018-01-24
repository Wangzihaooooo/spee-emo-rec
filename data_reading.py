import numpy as np
import os
import training_params
import scipy.io.wavfile as wf

def get_iemocap_sentences(dir):
  print('begin reading.....')
  iemocap_data = []
  category=read_category()
  for i in range(1, 6):
    path_to_parent = dir +'Session' + str(i) + '/sentences/wav'
    wav_dir_list = os.listdir(path_to_parent)#返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    for wav_dir in wav_dir_list:
        path_to_wavdir = os.path.join(path_to_parent, wav_dir)
        wav_file_list=os.listdir(path_to_wavdir) #files = [os.path.splitext(f)[0] for f in files]#分离文件名与扩展名,默认返回(fname,fextension)元组，可做分片操作
        for wav_file in wav_file_list:
            print(wav_file)
            framerate, samples = wf.read(os.path.join(path_to_wavdir, wav_file))
            fname=os.path.splitext(wav_file)[0]
            emotion = category[fname]['emotion']
            if emotion in training_params.available_emotions:
                sentences_data = {}
                sentences_data['id'] = fname
                sentences_data['emotion']= emotion
                sentences_data['signal'] = samples
                iemocap_data.append(sentences_data)

  return iemocap_data

def get_iemocap_dialog(dir):
    print('begin reading.....')
    iemocap_data = []
    category = read_category()
    for i in range(1, 6):
        path_to_wav = dir + 'Session' + str(i) + '/dialog/wav/'
        wav_file_list = os.listdir(path_to_wav)  # 返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
        for wav_file in wav_file_list:
            print(wav_file)
            framerate, samples = wf.read(os.path.join(path_to_wav, wav_file))
            fname = os.path.splitext(wav_file)[0]
            for sentence_name in category.keys():
                if fname in sentence_name:
                    emotion = category[sentence_name]['emotion']
                    start_time = category[sentence_name]['start_time']
                    end_time = category[sentence_name]['end_time']
                    if emotion in training_params.available_emotions:
                        dialog_data = {}
                        dialog_data['id'] = sentence_name
                        dialog_data['emotion'] = emotion
                        dialog_data['signal'] = stereo2mono(samples[int(start_time * framerate):int(end_time * framerate)])
                        iemocap_data.append(dialog_data)

    return iemocap_data

def read_category():
    category = {}
    for i in range(1, 6):
        parent_dir = training_params.path_to_iemocap + 'Session' + str(i) + '/dialog/EmoEvaluation/'
        dirlist = os.listdir(parent_dir)
        for file in dirlist:
            if file.startswith('Ses') & file.endswith('.txt'):
                f = open(parent_dir + file,
                         'r').read()  # 'r'读模式、'w'写模式、'a'追加模式、'b'二进制模式、'+'读/写模式 默认是读模式  read( )表示读取全部内容 readline( )表示逐行读取
                f = np.array(f.split('\n'))
                idx_n = np.arange(len(f))[f == '']  # 根据f=='' 判断该行是否为空行 相邻两个语音段文本信息以一段空行为分割线  获取各个语音段的开头行的下标
                for i in range(len(idx_n) - 2):  # 减去末尾两段空行
                    g = f[idx_n[i] + 1:idx_n[i + 1]]  # 一个语音段的完整信息
                    head = g[0].split('\t')
                    sentence_name = head[1]
                    emotion_lab = head[2]
                    start_time = float(head[0][head[0].find('[') + 1:head[0].find(' - ')])
                    end_time = float(head[0][head[0].find(' - ') + 3:head[0].find(']')])
                    category.update({sentence_name: {'emotion':emotion_lab,'start_time':start_time,'end_time':end_time}})

    return category

#根据key返回data对应的键值
def get_field(data, key):
    return np.array([e[key] for e in data])

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
        if "pk" in os.path.splitext(file)[1]:
            os.remove(file)

#立体声to单声道
def stereo2mono(x):
    '''
    This function converts the input signal (stored in a numpy array) to MONO (if it is STEREO)
    '''
    if isinstance(x, int):
        return -1
    if x.ndim==1:
        return x
    elif x.ndim==2:
        if x.shape[1]==1:
            return x.flatten()
        else:
            if x.shape[1]==2:
                return ( (x[:,1] / 2) + (x[:,0] / 2) )
            else:
                return -1