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
## loss曲线：
__epochs=25__
![epoch_25](epochs_25.png)

## librosa版本问题
这里使用的 *librosa==0.7.2*  
可能会出现  
![image](https://github.com/WThirteen/asr_thchs30/assets/100677199/6022f953-e40b-4b9e-9009-24a69d8a6e14)  
**参考这份博客：**

[解决不联网环境pip安装librosa、numba、llvmlite报错和版本兼容问题](https://blog.csdn.net/qq_39691492/article/details/130829401)  

*修改如下：*  

![image](https://github.com/WThirteen/asr_thchs30/assets/100677199/14ef3f58-7bb1-4f85-bc58-d49d761a86ae)
