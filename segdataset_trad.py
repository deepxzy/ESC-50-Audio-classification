import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd
# define directorie
# load a wave data

# # change wave data to mel-stft
# def calculate_melsp(x, n_fft=1024, hop_length=128):
#     stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
#     log_stft = librosa.power_to_db(stft)
#     melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
#     return melsp
# def calculate_melsp_2D(x, n_fft=1024, hop_length=128):
#     stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
#     log_stft = librosa.power_to_db(stft)
#     melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=1)
#     return melsp
# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()
# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()
# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))
# data augmentation: shift sound in timeframe
def shift_sound(x, rate=3):
    return np.roll(x, int(len(x)//rate))
# def read_file_list(root1=r"meta/esc50.csv",root2=r"audio",type='train',random_state=1):
#     meta_data = pd.read_csv(root1)
#
#     name = list(meta_data.loc[:, "filename"])
#     label = list(meta_data.loc[:, "target"])
#     path=[os.path.join(root2,i) for i in name]
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(path, label, test_size=0.2, random_state=random_state)
#     if type=='train':
#         return x_train, y_train
#     if type == 'test':
#         return x_test, y_test
def read_file_list(root1=r"meta/esc50.csv",root2=r"audio",type='train',random_state=1):
    meta_data = pd.read_csv(root1)

    name = list(meta_data.loc[:, "filename"])
    label = list(meta_data.loc[:, "target"])
    esc10 = list(meta_data.loc[:, "esc10"])
    label_ESC10=[0,1,10,11,12,20,21,38,40,41]
    path=[os.path.join(root2,i) for i in name]
    path1,label1=[],[]
    for i in range(len(path)):
        if esc10[i]:
            path1.append(path[i])
            label1.append(label_ESC10.index(label[i]))
    x_train, x_test, y_train, y_test = model_selection.train_test_split(path1, label1, test_size=0.2, random_state=1)
    if type=='train':
        return x_train, y_train
    if type == 'test':
        return x_test, y_test
def traditional_seg(root1=r"meta/esc50.csv",root2=r"audio",type='train',random_state=1):
    signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)
    features=[]
    for signal in signals:
        X, sample_rate= librosa.load(signal, sr=44100)
        stft = np.abs(librosa.stft(X))

        # mfcc (mel-frequency cepstrum)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=10).T, axis=0)
        # chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        # melspectrogram
        mel = np.mean(librosa.feature.melspectrogram(S=stft, n_mels=10, sr=sample_rate).T, axis=0)
        # spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        # print(mfccs.shape,chroma.shape,mel.shape,contrast.shape,tonnetz.shape)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features.append(ext_features)
    return features, labels

def traditional_tree(root1=r"meta/esc50.csv",root2=r"audio",type='train',random_state=1):
    signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)
    features=[]
    for signal in signals:
        X, sample_rate= librosa.load(signal, sr=44100)
        mean=np.mean(X)
        var=np.var(X)
        zeros=np.mean(librosa.zero_crossings(X))

        stft = np.abs(librosa.stft(X))

        # mfcc (mel-frequency cepstrum)

        mfccs = np.mean(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=10).T, axis=0))
        # chroma
        chroma = np.mean(np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0))
        # melspectrogram
        mel = np.mean(np.mean(librosa.feature.melspectrogram(S=stft, n_mels=10, sr=sample_rate).T, axis=0))
        # spectral contrast
        contrast = np.mean(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0))
        tonnetz = np.mean(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0))
        # # print(mfccs.shape,chroma.shape,mel.shape,contrast.shape,tonnetz.shape)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])#0.65
        # ext_features = np.hstack([mfccs, chroma, contrast, tonnetz])#0.6
        # ext_features = np.hstack([mfccs, chroma, mel, contrast])#0.675
        # # ext_features = np.hstack([mfccs,  mel, contrast])#0.5
        # # ext_features = np.hstack([mfccs, chroma,mel])#0.5625
        # ext_features = np.hstack([mfccs, chroma, mel, contrast, var])#0.7
        # ext_features = np.hstack([mfccs, chroma, mel, contrast, var])
        features.append(ext_features)
    return features, labels

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root1=r"meta/esc50.csv",root2=r"audio",type='train',random_state=1):

        signals, labels = read_file_list(root1=root1,root2=root2,type=type,random_state=random_state)
        # self.images = self.filter(images)  # images list
        # self.labels = self.filter(labels)  # labels list
        self.signals = signals  # images list
        self.labels = labels  # labels list

        print('Read ' + str(len(self.labels)) + ' valid examples')

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        X, sample_rate = librosa.load(signal, sr=44100)

        stft = np.abs(librosa.stft(X))


        # mfcc (mel-frequency cepstrum)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
        # chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        # melspectrogram
        mel = np.mean(librosa.feature.melspectrogram(S=stft, n_mels=20,sr=sample_rate).T, axis=0)
        # spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        ext_features = torch.from_numpy(ext_features).type(torch.FloatTensor)
        label = torch.tensor(label)
        return ext_features, label  # float32 tensor, uint8 tensor
#############################################
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':

    voc_train = traditional_seg()

