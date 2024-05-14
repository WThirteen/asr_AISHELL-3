#导入相关的库
from keras.models import Model
from keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import pickle
import glob
from tqdm import tqdm
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa
from IPython.display import Audio
import config


def load_texts_data(path,en_num_all):
    f=open(path, "r", encoding='utf-8')
    txt=[]
    for line in f:
        txt.append(line.strip())

    # 音频文件名字
    txt_filename = []
    # 对应文件内容
    txt_wav = []

    for i in txt:
        temp_txt_filename,temp_txt_wav = i.split('\t')
        txt_filename.append(temp_txt_filename)
        txt_wav.append(temp_txt_wav)

    # 音频文字
    texts = []

    for i in txt_wav:
        temp = ''
        for j in i:
            if j in en_num_all:
                continue
            else:
                # print(j)
                temp = temp+j
        texts.append(temp)
    
    return texts 

def create_en_num():
    # 字典 字母+数字
    en_num_all = []
    # 字母
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        en_num_all.extend(letter)
    # 数字
    for number in range(10):  
        en_num_all.extend(str(number))
    # 空格
    en_num_all.extend(' ')

    return en_num_all

def load_wav_data(path):
    files = os.listdir(path)
    # 音频文件
    wav_file = []
    # 音频文件的相对路径
    wav_file_path = []

    for i in files:
        temp_files = os.listdir(path+i)
        for j in temp_files:
            wav_file.append(j)
            wav_file_path.append(path+i+'/'+j)

    return wav_file_path

#根据数据集标定的音素读入
def load_and_trim(path):
    audio, sr = librosa.load(path)
    # energy = librosa.feature.rmse(audio)
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    return audio, sr

#可视化，显示语音文件的MFCC图
def visualize(paths,texts,index,mfcc_dim):
    path = paths[index]
    text = texts[index]
    print('Audio Text:', text)

    audio, sr = load_and_trim(path)
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()

    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    print('Shape of MFCC:', feature.shape)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_xticks(np.arange(0, 13, 2), minor=False);
    plt.show()

    return path

# Audio(visualize(0))

def wav_features(paths,total):
    #提取音频特征并存储
    features = []
    for i in tqdm(range(total)):
        path = paths[i]
        audio, sr = load_and_trim(path)
        features.append(mfcc(audio, sr, numcep=config.mfcc_dim, nfft=551))
    return features

def save_features(features):
    with open(config.features_path, 'wb') as fw:
        pickle.dump(features,fw)

def load_features():
    with open(config.features_path, 'rb') as f:  
        features = pickle.load(f)
    return features

def normalized_features(features): 
    #随机选择100个数据集
    samples = random.sample(features, 100)
    samples = np.vstack(samples)
    #平均MFCC的值为了归一化处理
    mfcc_mean = np.mean(samples, axis=0)
    #计算标准差为了归一化
    mfcc_std = np.std(samples, axis=0)
    # print(mfcc_mean)
    # print(mfcc_std)
    #归一化特征
    features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]

    return mfcc_mean,mfcc_std,features

def save_labels(texts):
    #将数据集读入的标签和对应id存储列表
    chars = {}
    for text in texts:
        for c in text:
            chars[c] = chars.get(c, 0) + 1

    chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
    chars = [char[0] for char in chars]
    # print(len(chars), chars[:100])

    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}

    return char2id,id2char

def data_set(total,features,texts):
    data_index = np.arange(total)
    np.random.shuffle(data_index)
    train_size = int(0.9 * total)
    test_size = total - train_size
    train_index = data_index[:train_size]
    test_index = data_index[train_size:]
    #神经网络输入和输出X,Y的读入数据集特征
    X_train = [features[i] for i in train_index]
    Y_train = [texts[i] for i in train_index]
    X_test = [features[i] for i in test_index]
    Y_test = [texts[i] for i in test_index]

    return X_train,Y_train,X_test,Y_test


#定义训练批次的产生，一次训练16个
def batch_generator(x, y,char2id):
    batch_size = config.batch_size
    offset = 0
    while True:
        offset += batch_size

        if offset == batch_size or offset >= len(x):
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            x = [x[i] for i in data_index]
            y = [y[i] for i in data_index]
            offset = batch_size

        X_data = x[offset - batch_size: offset]
        Y_data = y[offset - batch_size: offset]

        X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
        Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])

        X_batch = np.zeros([batch_size, X_maxlen, config.mfcc_dim])
        Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
        X_length = np.zeros([batch_size, 1], dtype='int32')
        Y_length = np.zeros([batch_size, 1], dtype='int32')

        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, :X_length[i, 0], :] = X_data[i]

            Y_length[i, 0] = len(Y_data[i])
            Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]

        inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
        outputs = {'ctc': np.zeros([batch_size])}

        yield (inputs, outputs)

def input_layer():
    X = Input(shape=(None, config.mfcc_dim,), dtype='float32', name='X')
    Y = Input(shape=(None,), dtype='float32', name='Y')
    X_length = Input(shape=(1,), dtype='int32', name='X_length')
    Y_length = Input(shape=(1,), dtype='int32', name='Y_length')

    return X,Y,X_length,Y_length


#卷积1层
def conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None,
                  dilation_rate=dilation_rate)(inputs)

#标准化函数
def batchnorm(inputs):
    return BatchNormalization()(inputs)

#激活层函数
def activation(inputs, activation):
    return Activation(activation)(inputs)

#全连接层函数
def res_block(inputs, filters, kernel_size, dilation_rate):
    hf = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
    hg = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
    h0 = Multiply()([hf, hg])

    ha = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
    hs = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')

    return Add()([ha, inputs]), hs

#计算损失函数
def calc_ctc_loss(args):
    y, yp, ypl, yl = args
    return K.ctc_batch_cost(y, yp, ypl, yl)

def model_train(X,Y,X_length,Y_length,char2id):  
    h0 = activation(batchnorm(conv1d(X, config.filters, 1, 1)), 'tanh')
    shortcut = []
    for i in range(config.num_blocks):
        for r in [1, 2, 4, 8, 16]:
            h0, s = res_block(h0, config.filters, 7, r)
            shortcut.append(s)

    h1 = activation(Add()(shortcut), 'relu')
    h1 = activation(batchnorm(conv1d(h1, config.filters, 1, 1)), 'relu')
    #softmax损失函数输出结果
    Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
    sub_model = Model(inputs=X, outputs=Y_pred)

    ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
    #加载模型训练
    model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
    #建立优化器
    optimizer = SGD(lr=0.02, momentum=0.9, nesterov=True, clipnorm=5)
    #激活模型开始计算
    model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)
    
    return sub_model,model

def save_model_pkl(sub_model,char2id,id2char,mfcc_mean,mfcc_std):
    #保存模型
    sub_model.save(config.model_path)
    #将字保存在pl=pkl中
    with open(config.pkl_path, 'wb') as fw:
        pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)


def draw_loss(history):
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    plt.plot(np.linspace(1, config.epochs, config.epochs), train_loss, label='train')
    plt.plot(np.linspace(1, config.epochs, config.epochs), valid_loss, label='valid')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def run():
    print("-----load data-----")
    path_train = load_wav_data(path=config.train_wav_data_path)
    path_test = load_wav_data(path=config.test_wav_data_path)
    paths = []
    paths.extend(path_train), paths.extend(path_test)

    privacy_dict = create_en_num()
    texts_train = load_texts_data(path=config.train_texts_data_path, en_num_all=privacy_dict)
    texts_test = load_texts_data(path=config.test_texts_data_path, en_num_all=privacy_dict)
    texts = []
    texts.extend(texts_train), texts.extend(texts_test)

    char2id,id2char = save_labels(texts)

    total = len(texts)

    # print("-----Extract audio features-----")
    # features = wav_features(paths,total)

    # print("-----save features-----")
    # save_features(features)

    print("-----load features-----")
    features = load_features()

    
    mfcc_mean,mfcc_std,features = normalized_features(features)

    X_train,Y_train,X_test,Y_test = data_set(total,features,texts)

    X,Y,X_length,Y_length = input_layer()
    
    sub_model,model = model_train(X,Y,X_length,Y_length,char2id)
    
    # 回调
    checkpointer = ModelCheckpoint(filepath=config.model_path, verbose=0)
    # 监控 损失值（loss）作为指标
    lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=1e-6)
    #开始训练
    history = model.fit_generator(
        generator=batch_generator(X_train, Y_train, char2id),
        steps_per_epoch=len(X_train) // config.batch_size,
        epochs=config.epochs,
        validation_data=batch_generator(X_test, Y_test, char2id),
        validation_steps=len(X_test) // config.batch_size,
        callbacks=[checkpointer, lr_decay])
    
    save_model_pkl(sub_model,char2id,id2char,mfcc_mean,mfcc_std)
    draw_loss(history)


if __name__ == '__main__' :
    run()
