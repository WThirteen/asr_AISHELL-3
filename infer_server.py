import argparse
import functools
import os

import torch
import uvicorn
from fastapi import FastAPI, File, Body, UploadFile, Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from utils.utils import add_arguments, print_arguments

from keras.models import load_model
from keras import backend as K
import numpy as np
import librosa
from python_speech_features import mfcc
import speech_recognition as sr
import pickle
import config
import wave
import io  

from pydub import AudioSegment  
from io import BytesIO  
import librosa  
import numpy as np  
from python_speech_features import mfcc  

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",        type=str,  default="0.0.0.0",   help="监听主机的IP地址")
add_arg("port",        type=int,  default=5000,        help="服务所使用的端口号")
# add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
# add_arg("model_path",  type=str,  default="models/tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("num_beams",   type=int,  default=1,      help="解码搜索大小")
add_arg("batch_size",  type=int,  default=16,     help="预测batch_size大小")
add_arg("use_compile", type=bool, default=False,  help="是否使用Pytorch2.0的编译器")
add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
args = parser.parse_args()
print_arguments(args)

# 设置设备
device = "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32


model = load_model(config.model_path)

with open(config.pkl_path, 'rb') as fr:
        [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)


app = FastAPI(title="thirteen语音识别")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")
model_semaphore = None


def release_model_semaphore():
    model_semaphore.release()


def recognition(file: File,mfcc_mean, mfcc_std):

    X_data = extract_mfcc_features(file, mfcc_mean, mfcc_std)
    pred = model.predict(np.expand_dims(X_data, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()
    results = ''
    judge=0
    for i in pred_ids:
        if i != -1:
            judge=1
            results = results + id2char[i]
    if judge!=1:
        results = '未检测到'

    return results


def extract_mfcc_features(audio_bytes,  mfcc_mean, mfcc_std):  
    # 使用pydub将bytes转换为WAV格式的AudioSegment（如果它不是WAV的话）  
    # 注意：这里我们假设input_bytes是WAV或我们可以转换为WAV的格式  
    # 如果input_bytes不是WAV且格式未知，你可能需要先检测它  
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")  # 如果已经是WAV，或者确定可以解析为WAV  
 
    # 确保输出为WAV格式（如果之前不是的话，这一步其实是多余的，因为from_file已经处理了）  
    # 但为了清晰起见，我们还是将其导出为WAV的bytes  
    wav_bytes = BytesIO()  
    audio_segment.export(wav_bytes, format="wav")  
 
    # 重置BytesIO的指针到开头  
    wav_bytes.seek(0)  
 
    # 使用librosa加载WAV音频  
    y, sr = librosa.load(wav_bytes)  
 
    # 提取RMS能量  
    energy = librosa.feature.rms(y=y)  
 
    # 找到能量大于最大能量1/5的帧  
    frames = np.nonzero(energy[0] >= np.max(energy[0]) / 5)  
 
    # 将帧索引转换为样本索引  
    if frames[0].size:  
        indices = librosa.core.frames_to_samples(frames)[0]  
        y = y[indices[0]:indices[-1]]  
 
    # 提取MFCC特征  
    mfcc_dim = 13  # 你可以根据需要修改MFCC的维度  
    mfcc_features = mfcc(y, sr, numcep=mfcc_dim, nfft=551)  
 
    # 这里假设你已经有了mfcc_mean和mfcc_std用于标准化（通常需要在训练阶段计算）  
    # 如果没有，你可以跳过标准化步骤，或者计算它们  
    mfcc_features = (mfcc_features - mfcc_mean) / (mfcc_std + 1e-14)  

    return mfcc_features  


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="音频文件")):
    # if language == "None": language = None
    data = await audio.read()
    with io.BytesIO(data) as bio:
        with wave.open(bio, 'rb') as wav_file:
            pass
    results = recognition(file= data, mfcc_mean= mfcc_mean, mfcc_std= mfcc_std)
    ret = {"results": results, "code": 0}
    return ret


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)
