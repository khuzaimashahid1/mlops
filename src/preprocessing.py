import os
import librosa
import numpy as np
import soundfile as sf
import argparse
from typing import Text
import yaml

def segmenter(input_dir,sr,db_level):
    # this code only works when the audio duration is greater than two seconds given that cough amplitude goes above
    # 50db threshold.
    segments=[]
    count=0
    audio_waves = np.load(input_dir,allow_pickle=True)
    for aud in audio_waves:
        try:
            if len(aud)/sr>1.0:
                splits=librosa.effects.split(aud,top_db=db_level)
                print(splits)
                for sp in splits:
                    pads=sp[1]-sp[0]
                    if pads<44100:
                        diff=44100-pads
                        zeros=np.zeros(int(diff/2))
                        tsec=np.pad(aud[sp[0]:sp[1]],len(zeros))
                        tsec = librosa.util.normalize(tsec)
                        segments.append(tsec)
                        count+=1
        except:
            print('file length zero')
    
    return segments

def main(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    pos = segmenter(
        config['preprocessing']['raw_audio_pos'],
        config['audio_&_feature']['sampling_rate'],
        config['preprocessing']['db_level'],
        )
    
    neg = segmenter(
        config['preprocessing']['raw_audio_neg'],
        config['audio_&_feature']['sampling_rate'],
        config['preprocessing']['db_level'],
        )
    
    np.save(config['preprocessing']['processed_audio_pos'],pos)
    np.save(config['preprocessing']['processed_audio_neg'],neg)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)

