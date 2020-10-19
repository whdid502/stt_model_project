from mfcc_fbank import MFCC, FilterBank
import os

os.chdir(r'C:\Users\whdid\OneDrive\바탕 화면\stt_project\feature_extraction\sound_data')

# mfcc_test = MFCC('KsponSpeech_123997.pcm')

fbank_test = FilterBank('KsponSpeech_123997.pcm')