import os, glob, ntpath, shutil, numpy
import scipy.io.wavfile as wavfile
import pydub
#import eyed3
import aifc
from pydub import AudioSegment

def convertFsDirWavToWav(dirName, Fs, nC):
    '''
    This function converts the WAV files stored in a folder to WAV using a different sampling freq and number of channels.
    ARGUMENTS:
     - dirName:     the path of the folder where the WAVs are stored
     - Fs:          the sampling rate of the generated WAV files
     - nC:          the number of channesl of the generated WAV files
    '''

    types = (dirName+os.sep+'*.wav',) # the tuple of file types
    filesToProcess = []

    for files in types:
        filesToProcess.extend(glob.glob(files))     

    newDir = dirName + os.sep + "Fs" + str(Fs) + "_" + "NC"+str(nC)
    if os.path.exists(newDir) and newDir!=".":
        shutil.rmtree(newDir)   
    os.makedirs(newDir) 

    for f in filesToProcess:    
        _, wavFileName = ntpath.split(f)    
        command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + newDir + os.sep + wavFileName + "\"";
        print (command)
        os.system(command)

def readAudioFile(path):
    '''
    This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file
    '''
    extension = os.path.splitext(path)[1]#分离文件名与扩展名,默认返回(fname,fextension)元组，可做分片操作

    try:
        #if extension.lower() == '.wav':
            #[Fs, x] = wavfile.read(path)
        if extension.lower() == '.aif' or extension.lower() == '.aiff':
            s = aifc.open(path, 'r')
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            x = numpy.fromstring(strsig, numpy.short).byteswap()
            Fs = s.getframerate()
        elif extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au':            
            try:
                audiofile = AudioSegment.from_file(path)
            #except pydub.exceptions.CouldntDecodeError:
            except:
                print (		"Error: file not found or other I/O error. (DECODING FAILED)")
                return (-1,-1)                

            if audiofile.sample_width==2:                
                data = numpy.fromstring(audiofile._data, numpy.int16)
            elif audiofile.sample_width==4:
                data = numpy.fromstring(audiofile._data, numpy.int32)
            else:
                return (-1, -1)
            Fs = audiofile.frame_rate
            x = []
            for chn in range(audiofile.channels):
                x.append(data[chn::audiofile.channels])
            x = numpy.array(x).T
        else:
            print ("Error in readAudioFile(): Unknown file type!")
            return (-1,-1)
    except IOError: 
        print ("Error: file not found or other I/O error.")
        return (-1,-1)

    if x.ndim==2:
        if x.shape[1]==1:
            x = x.flatten()

    return (Fs, x)

def stereo2mono(x):
    '''
    立体声to单声道
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

