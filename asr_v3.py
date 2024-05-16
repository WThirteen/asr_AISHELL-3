from keras.models import load_model
from keras import backend as K
import numpy as np
import librosa
from python_speech_features import mfcc
import speech_recognition as sr
import pickle
import glob
import config
import wave
import os
import pyaudio
from tqdm import tqdm

class set_audio():
    CHUNK = 1024  # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16  # 采样位数
    CHANNELS = 1  # 单声道
    RATE = 44100  # 采样频率

# 可设置录制时间
def record_audio(record_second):
    """ 录音功能 """
    p = pyaudio.PyAudio()  # 实例化对象
    stream = p.open(format=set_audio.FORMAT,
                    channels=set_audio.CHANNELS,
                    rate=set_audio.RATE,
                    input=True,
                    frames_per_buffer=set_audio.CHUNK)  # 打开流，传入响应参数
    
    wf = wave.open('temp_file.wav', 'wb')  # 打开 wav 文件。
    wf.setnchannels(set_audio.CHANNELS)  # 声道设置
    wf.setsampwidth(p.get_sample_size(set_audio.FORMAT))  # 采样位数设置
    wf.setframerate(set_audio.RATE)  # 采样频率设置

    for _ in tqdm(range(0, int(set_audio.RATE * record_second / set_audio.CHUNK))):
        data = stream.read(set_audio.CHUNK)
        wf.writeframes(data)  # 写入数据
    stream.stop_stream()  # 关闭流
    stream.close()
    p.terminate()
    wf.close()

    wavs = glob.glob('temp_file.wav')
    
    # os.remove("temp_file.wav")

    return wavs



def save_as_wav(audio, output_file_path):
    with wave.open(output_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位PCM编码
        wav_file.setframerate(44100)  # 采样率为44.1kHz
        wav_file.writeframes(audio.frame_data)

# 录音自动停止
def input_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说...")
        r.pause_threshold = 1
        audio = r.listen(source)
        output_file_path = "temp_file.wav"
        save_as_wav(audio, output_file_path)
        wavs = glob.glob('temp_file.wav')
        
    return wavs

def load_file():
    with open(config.pkl_path, 'rb') as fr:
        [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)
    model = load_model(config.model_path)
    return char2id, id2char, mfcc_mean, mfcc_std, model

def set_data(wavs, mfcc_mean, mfcc_std):
    mfcc_dim = config.mfcc_dim
    index = np.random.randint(len(wavs))
    audio, sr = librosa.load(wavs[index])
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)
    # print(X_data.shape)
    return X_data


def wav_pred(model,X_data,id2char):
    pred = model.predict(np.expand_dims(X_data, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()
    words=''
    judge=0
    for i in pred_ids:
        if i != -1:
            judge=1
            words=words+id2char[i]
    if judge==1:
        print(words)
    else:
        print("未检测到")

def run():
    # 自动停止录音
    # wavs = input_audio()
    # 设置录制时间
    wavs = record_audio(record_second=5)
    char2id, id2char, mfcc_mean, mfcc_std, model = load_file()
    X_data = set_data(wavs, mfcc_mean, mfcc_std)
    wav_pred(model,X_data,id2char)
    os.remove("temp_file.wav")


if __name__ == '__main__' :
    run()

