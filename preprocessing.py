import os
import librosa
import numpy as np
import soundfile as sf

class cough_segmentation():
    def segmenter(self,input_dir,output_dir):
        # this code only works when the audio duration is greater than two seconds given that cough amplitude goes above
        # 50db threshold.

        count=0
        for i in os.listdir(input_dir):
            try:
                aud, sr = librosa.load(input_dir+i,sr=22050)
                aud = librosa.util.normalize(aud)

                if len(aud)/sr>1.0:
                    splits=librosa.effects.split(aud,top_db=33)
                    print(splits)
                    for sp in splits:
                        pads=sp[1]-sp[0]
                        if pads<44100:
                            diff=44100-pads
                            zeros=np.zeros(int(diff/2))
                            tsec=np.pad(aud[sp[0]:sp[1]],len(zeros))
                            tsec = librosa.util.normalize(tsec)
                            n=i.split('.')[0]
                            sf.write(f'{output_dir}{n+"_"+str(count)}.wav',tsec ,sr,format='wav')
                            count+=1
            except:
                print('file length zero')



