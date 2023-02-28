import librosa
import numpy as np
import os
import pandas as pd

# random
# librosa features column names formation
def librosa_feature_columns():
#     1 1 1 1 1 40 128 12 12 12 7 6
    headers=[]
    mf=['mfcc'+str(i) for i in range(40)]
    mspec=['melspec'+str(i) for i in range(40)]
    ch_sftf=['chroma_stft'+str(i) for i in range(36)]
    chr_cq=['chroma_cq'+str(i) for i in range(36)]
    chcens=['chroma_cens'+str(i) for i in range(36)]
    const=['contrast'+str(i) for i in range(7)]
    tonn=['tonnetz'+str(i) for i in range(6)]
    [headers.append(i) for i in ['rms0','spec_cent0','spec_bw0','rolloff0','zcr0']]
    [headers.append(i) for i in mf]
    [headers.append(i) for i in mspec]
    [headers.append(i) for i in ch_sftf]
    [headers.append(i) for i in chr_cq]
    [headers.append(i) for i in chcens]
    [headers.append(i) for i in const]
    [headers.append(i) for i in tonn]
    # [headers.append(i) for i in ['filename']]

    # dd.columns=headers
    return headers

def librosa_most_important_features(dir):

    als=[]
    alss=[]

    adata=[i for i in os.listdir(dir) if i.split('.')[1]=='wav' or i.split('.')[1]=='mp3'
        or i.split('.')[1]=='flac' or i.split('.')[1]=='m4a' ]

    positive=[]
    for i in sorted(adata):
        positive=[]
        als=[]
        y, sr = librosa.load(dir+i,sr=22050)
        y = librosa.util.normalize(y)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T,axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=36).T,axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=36).T,axis=0)
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        positive.append([spec_bw])
        positive.append(mfcc)
        positive.append(chroma_stft)
        positive.append(chroma_cens)
        positive.append(contrast)

        pos=np.array(positive)

        for z in pos:
            for x in z:
                als.append(x)
                
        als.append(i)
        alss.append(als)

    return alss

# librosa features
def librosa_normal_embedding(y,sr):

    embedding=[]
    
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    melspec= np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T,axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=36).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=36).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=36).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr,).T,axis=0)
    
    embedding=[rms,spec_cent,spec_bw,rolloff,zcr,
               *mfcc,*melspec,*chroma_stft,*chroma_cq,*chroma_cens,*contrast,*tonnetz]
    
    return embedding