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

def save_as_wav(audio, output_file_path):
    with wave.open(output_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位PCM编码
        wav_file.setframerate(44100)  # 采样率为44.1kHz
        wav_file.writeframes(audio.frame_data)

def input_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说...")
        r.pause_threshold = 1
        audio = r.listen(source)
        output_file_path = "temp_file.wav"
        save_as_wav(audio, output_file_path)
        wavs = glob.glob('temp_file.wav')
        os.remove("temp_file.wav")
        return wavs

def out_load_audio():
    path = config.audio_path
    wavs = glob.glob(path)
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
    print(X_data.shape)
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
    # wavs = input_audio()
    wavs = out_load_audio()
    char2id, id2char, mfcc_mean, mfcc_std, model = load_file()
    X_data = set_data(wavs, mfcc_mean, mfcc_std)
    wav_pred(model,X_data,id2char)
    


if __name__ == '__main__' :
    run()

