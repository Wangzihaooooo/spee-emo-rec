import numpy as np
import os
import training_params
import scipy.io.wavfile as wf
import librosa

def get_chinese_dataset(dir):
    print('begin reading.....')
    iemocap_data=[]
    names_list = os.listdir(dir)  # 返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    for name_dir in names_list:
        path_to_namedir=os.path.join(dir, name_dir)
        category_dir_list=os.listdir(path_to_namedir)
        for category_dir in category_dir_list:
            path_to_categorydir = os.path.join(path_to_namedir, category_dir)
            wav_file_list = os.listdir(path_to_categorydir)
            for wav_file in wav_file_list:
                framerate, samples = wf.read(os.path.join(path_to_categorydir, wav_file))
                fname = os.path.splitext(wav_file)[0]
                emotion = category_dir[:3]
                if emotion in training_params.available_emotions:
                    print(name_dir + '_' + category_dir + '_' + fname)
                    dataset_data = {}
                    dataset_data['id'] = name_dir+'_'+category_dir+'_'+fname
                    dataset_data['emotion'] = emotion
                    dataset_data['signal'] = samples
                    dataset_data['framerate'] = framerate
                    iemocap_data.append(dataset_data)

    return np.array(iemocap_data)

def get_iemocap_sentences(dir):
  print('begin reading.....')
  iemocap_data = []
  category=read_category()
  num = {'ang': 0, 'neu': 0, 'sad': 0, 'hap': 0, 'exc': 0, 'sur': 0,'fru': 0}
  #{'ang': 1090, 'neu': 1704, 'sad': 1077, 'hap': 595, 'exc': 1041, 'sur': 105, 'fru': 1829}
  for i in range(1, 6):
    path_to_parent = dir +'Session' + str(i) + '/sentences/wav'
    wav_dir_list = os.listdir(path_to_parent)#返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    for wav_dir in wav_dir_list:
        path_to_wavdir = os.path.join(path_to_parent, wav_dir)
        wav_file_list=os.listdir(path_to_wavdir) #files = [os.path.splitext(f)[0] for f in files]#分离文件名与扩展名,默认返回(fname,fextension)元组，可做分片操作
        for wav_file in wav_file_list:
            fname = os.path.splitext(wav_file)[0]
            emotion = category[fname]['emotion']
            if emotion in training_params.available_emotions:
                num[emotion] = num[emotion] + 1
                print(emotion+'_'+wav_file)
                #samples ,framerate = librosa.load(os.path.join(path_to_wavdir, wav_file),sr=16000)
                framerate, samples = wf.read(os.path.join(path_to_wavdir, wav_file))
                sentences_data = {}
                sentences_data['id'] = emotion+'_'+fname
                sentences_data['emotion']= emotion
                sentences_data['signal'] = samples
                sentences_data['framerate'] = framerate
                iemocap_data.append(sentences_data)
  print(num)

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
                        print(emotion + '_' + wav_file)
                        dialog_data = {}
                        dialog_data['id'] = emotion+'_'+sentence_name
                        dialog_data['emotion'] = emotion
                        dialog_data['signal'] = stereo2mono(samples[int(start_time * framerate):int(end_time * framerate)])
                        dialog_data['framerate'] = framerate
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
        if "." in os.path.splitext(file)[0]:
            os.remove(file)
        if "pk" in os.path.splitext(file)[1]:
            os.remove(file)
        if "lab" in os.path.splitext(file)[1]:
            os.remove(file)
        if "peak" in os.path.splitext(file)[1]:
            os.remove(file)
        if "tag" in os.path.splitext(file)[1]:
            os.remove(file)
        if "ini" in os.path.splitext(file)[1]:
            os.remove(file)
#process_iemocap_data(training_params.path_to_chinese_dataset)
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