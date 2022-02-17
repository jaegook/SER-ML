import librosa
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imsave

from scipy.fft import fft
from librosa import display

#all audio handling functions

def load_audio_file(filepath, sr):
   return librosa.load(filepath, sr=sr)

def wav_to_spectrum(wav):
   return np.abs(fft(wav))

def wav_to_mel(wav, sr, n_fft, hop_length, n_mels):
   #print("in wav_to_mel")
   #print("type(wav)", type(wav))
   #print("wav.shape=", wav.shape)
   #print("sr=", sr)

   mel_spectrogram = librosa.feature.melspectrogram(wav, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
   
   #print("type(mel_spectrogram)=", type(mel_spectrogram))
   
   log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
   
   #print("typpe(log_mel_spectrogram)=", type(log_mel_spectrogram))
   
   return log_mel_spectrogram

def plot_time_domain(wav, sr):
   plt.figure()
   librosa.display.waveplot(y = wav, sr = sr)
   plt.xlabel("Time (seconds) -->")
   plt.ylabel("Amplitude")
   plt.savefig("./images/time_dom.png")

def plot_freq_domain(spectrum, sr):
   librosa.display.waveplot(y = spectrum, sr=sr)
   plt.savefig("./images/freq_dom.png")
   

def plot_spectrum(log_mel_spectrogram, sr):
   librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
   plt.savefig("./images/log_mel_spectrogram.png")
   


#fn that gets number of samples given sr, length of audio
#fn that gets length given sr and number of samples


def test(filepath):
   wav, sr = load_audio_file(filepath)
   print("type(wav)=", type(wav))
   print("wav =", wav)
   print("wav.shape=", wav.shape)
   print("sr =", sr)
   
   y = np.arange(wav.shape[0])
   plot_time_domain(wav, sr)
   
   spectrum = wav_to_spectrum(wav)
   print("type(spectrum)=", type(spectrum))
   print("spectrum=", spectrum)
   print("spectrum.shape=", spectrum.shape)
   plot_freq_domain(spectrum, sr)

   log_mel_spectrogram = wav_to_mel(wav, sr, 2048, 512, 80)
   print("type(log_mel_spectrogram)=", type(log_mel_spectrogram))
   print("log_mel_spectrogram=", log_mel_spectrogram)
   print("log_mel_spectrogram.shape=", log_mel_spectrogram.shape)

   plot_spectrum(log_mel_spectrogram, sr)

def save_image(filename, x,):
   # use skimage and save numpy array to image
   imsave(filename, x)   
   
   
      
   
