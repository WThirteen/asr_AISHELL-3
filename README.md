# asr_AISHELL-3
使用AISHELL-3 数据集 训练语音识别模型

## 使用方法
创建虚拟环境
```
conda create -n asr python=3.10
```
配置环境
```
activate asr
pip install -r requirements.txt
```
运行
```
python train.py
```
如果已经运行且已经有音频特征文件 features.pkl  
可直接运行 __trian_v2.py__
```
python train_v2.py
```
* 在train_v4中增加tensorboard  
可查看训练日志
## 使用web进行语音识别
运行
```
python asr_web.py
```
* 可进行读取录音
* 本地录制并上传进行识别
* 预览
![image](https://github.com/WThirteen/asr_AISHELL-3/assets/100677199/59201975-12ea-46cf-9e4a-e490c02211c0)

## 查看训练日志
输入命令
```
tensorboard --logdir= log_path
```
## loss曲线：
__epochs=25__  
![epochs_25](https://github.com/WThirteen/asr_AISHELL-3/assets/100677199/c4ad5342-aee6-4950-833d-59c424b15f1e)

## librosa版本问题
这里使用的 *librosa==0.7.2*  
可能会出现  
![image](https://github.com/WThirteen/asr_thchs30/assets/100677199/6022f953-e40b-4b9e-9009-24a69d8a6e14)  
**参考这份博客：**

[解决不联网环境pip安装librosa、numba、llvmlite报错和版本兼容问题](https://blog.csdn.net/qq_39691492/article/details/130829401)  

*修改如下：*  

![image](https://github.com/WThirteen/asr_thchs30/assets/100677199/14ef3f58-7bb1-4f85-bc58-d49d761a86ae)

## api的部分参考
[Tensorflow-FaceRecognition](https://github.com/yeyupiaoling/Tensorflow-FaceRecognition)  
将原来微调的whisper模型换成这里训练的asr模型
