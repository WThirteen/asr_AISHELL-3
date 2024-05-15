# 训练数据音频文件路径
train_wav_data_path = 'AISHELL-3/train/wav/'

# 训练数据内容文件路径
train_texts_data_path = 'AISHELL-3/train/content.txt'

# 测试数据音频文件路径
test_wav_data_path = 'AISHELL-3/test/wav/'

# 测试数据内容文件路径
test_texts_data_path = 'AISHELL-3/test/content.txt'

# 存放模型路径 /模型名字
model_path = 'model/asr_AISHELL.h5'

# 存放pkl路径 /pkl名字
pkl_path = 'pkl_all/dictionary.pkl'

# 存放labels路径
labels_path = 'pkl_all/labels.pkl'

# features.pkl路径
features_path = 'pkl_all/features.pkl'

# 外部导入音频路径
audio_path = 'AISHELL-3/train/wav/SSB0005/SSB00050001.wav'
# model_name

batch_size = 16

epochs = 25

num_blocks = 3

filters = 128

mfcc_dim = 13
