import librosa
import numpy as np
import os
import pandas as pd
from typing import Text
import yaml
import argparse

# librosa features column names formation
def librosa_feature_columns(config):
#     1 1 1 1 1 40 128 12 12 12 7 6
    headers=[]
    mf=['mfcc'+str(i) for i in range(config['audio_&_feature']['n_mfcc'])]
    mspec=['melspec'+str(i) for i in range(config['audio_&_feature']['n_mels'])]
    ch_sftf=['chroma_stft'+str(i) for i in range(config['audio_&_feature']['n_chroma'])]
    chr_cq=['chroma_cq'+str(i) for i in range(config['audio_&_feature']['n_chroma'])]
    chcens=['chroma_cens'+str(i) for i in range(config['audio_&_feature']['n_chroma'])]
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

# librosa features
def librosa_normal_embedding(y,sr,config):

    # with open(config_path) as conf_file:
    #     config = yaml.safe_load(conf_file)

    embedding=[]
    
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config['audio_&_feature']['n_mfcc']).T,axis=config['audio_&_feature']['axis'])
    melspec= np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config['audio_&_feature']['n_mels'], fmax=config['audio_&_feature']['fmax']).T,axis=config['audio_&_feature']['axis'])
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=config['audio_&_feature']['n_chroma']).T,axis=config['audio_&_feature']['axis'])
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=config['audio_&_feature']['n_chroma']).T,axis=config['audio_&_feature']['axis'])
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=config['audio_&_feature']['n_chroma']).T,axis=config['audio_&_feature']['axis'])
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T,axis=config['audio_&_feature']['axis'])
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr,).T,axis=config['audio_&_feature']['axis'])
    
    embedding=[rms,spec_cent,spec_bw,rolloff,zcr,
               *mfcc,*melspec,*chroma_stft,*chroma_cq,*chroma_cens,*contrast,*tonnetz]
    
    return embedding

def feature_extractor(sr,embedding,data,columns,y,config):
    feats=[embedding(x,sr,config) for x in data]
    feats_df=pd.DataFrame(feats,columns=columns)
    ys=feats_df.shape[0]*y
    feats_df['status']=ys
    return feats_df

def main_caller(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    pos = np.load(config['preprocessing']['processed_audio_pos'],allow_pickle=True)
    neg = np.load(config['preprocessing']['processed_audio_neg'],allow_pickle=True)

    positive=['covid-19']
    negative=['healthy']

    columns = librosa_feature_columns(config)

    pos_features = feature_extractor(
        config['audio_&_feature']['sampling_rate'],
        librosa_normal_embedding,
        pos,
        columns,
        positive,
        config
        )
    
    neg_features = feature_extractor(
        config['audio_&_feature']['sampling_rate'],
        librosa_normal_embedding,
        neg,
        columns,
        negative,
        config
        )
    
    covid_features = pd.concat([pos_features,neg_features],axis=0).reset_index(drop=True)
    covid_features.to_csv(config['feature_extraction']['features_path'],index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main_caller(config_path=args.config)
