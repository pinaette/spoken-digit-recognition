import os
import time
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt

def human_readable_time(duration):
    '''
    function to return human-readable time as string (%h %m %s)
    '''
    if duration > 60:
        minutes = round(duration / 60)
        seconds = round(duration % 60)
        if minutes > 60:
            hours = round(minutes / 60)
            minutes = minutes % 60
        else:
         hours = 0
    else:
        hours = 0
        minutes = 0
        seconds = round(duration)
    return (str(hours)+'h', str(minutes)+'m', str(seconds)+'s')

def plot_mel_from_wav(path, filename, rate=None):
    '''
    load audio as waveform, extract sampling rate, convert both into melspectrogram, calculate decibels, plot.

    waveform is a time series, represented as a 1d numpy floating point array.
    by default .load() samples at 22050 (number of samples per second of audio) and mixes to mono.
    if rate is given, file is sampled at given rate instead of 22050.
    '''
    if rate:
        waveform, sampling_rate = librosa.load(path+filename, sr=rate)
    else:
        waveform, sampling_rate = librosa.load(path+filename)
    
    # create melspectrogram using waveform and sampling rate
    S = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
    # calculate decibels
    S_dB = librosa.power_to_db(S)

    # create plot
    fig, ax = plt.subplots()
    img = ld.specshow(S_dB, sr=sampling_rate, ax=ax)

    # save images as .png in imgs directory
    plt.savefig('imgs/' + filename[:-3] + 'png')
    plt.close()

# create relative directory within current directory to store images
os.mkdir('imgs')
# use relative directory to get list of filenames as string
filepath='data/'
# filepath='sample_data/'

files = os.listdir(filepath)

# get time before starting image creation
tick = time.time()
# store mel spectrograms as images
for file in files:
    plot_mel_from_wav(path=filepath, filename=file)
# get time after image creation is done
duration = time.time() - tick

# display elapsed time
print('elapsed time:', human_readable_time(duration))
