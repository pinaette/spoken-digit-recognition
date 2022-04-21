import librosa
import librosa.display as ld
import numpy as np
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

def load_audio(path, filename, sr=None):
    '''
    load audio as waveform and extract sampling rate.

    waveform is a time series, represented as a 1d numpy floating point array.
    by default .load() samples at 22050 (number of samples per second of audio) and mixes to mono.
    when sampling_rate (sr) set to None, keeps the original sampling rate of the file (8kHz in this case)

    for higher quality files, sr can be set to lower number:
    e.g. 8000 for 8kHz works well in this case and it's computationally efficient
    '''
    waveform, sampling_rate = librosa.load(path+filename)
    return waveform, sampling_rate


def oversample(original):
    '''
    creates more data by oversampling. 
    takes original waveform (np array) as input.
    returns original sample, as well as three augmented versions:
    different pitch and speed, increased amplitude, with noise
    '''
    # oversample by changing pitch and speed
    pitch_speed = original.copy()
    length_change = np.random.uniform(low=0.8, high = 1.0)
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(pitch_speed),speed_fac),np.arange(0,len(pitch_speed)),pitch_speed)
    minlen = min(pitch_speed.shape[0], tmp.shape[0])
    pitch_speed *= 0
    pitch_speed[0:minlen] = tmp[0:minlen]

    # oversample by value augmentation (increase amplitude)
    augmented = original.copy()
    dyn_change = np.random.uniform(low=1.5,high=3)
    augmented = augmented * dyn_change

    # oversample by adding noise distribution
    noisy = original.copy()
    noise_amp = 0.005*np.random.uniform()*np.amax(noisy)
    noisy = noisy.astype('float64') + noise_amp * np.random.normal(size=noisy.shape[0])

    return pitch_speed, augmented, noisy


def plot_mel_from_waveform(waveform, sampling_rate, filename, modifier=''):
    '''
    convert waveform into melspectrogram, calculate decibels, plot.
    takes in waveform and sampling rate to create a mel spectrogram.
    saves created image as .png to imgs/ directory (directory must be created before running this).
    
    `modifier` argument is for distinguishing between original and oversampled versions.
    if not provided at time of running the function and if oversampled copies are used
     as part of the same step in a for loop as the original, later saves will be overwritten and
     only the latest version of the img file (either the original or one of the augmented ones, depending
     on how the data is fed) will be saved, losing all others since all 4 have the same original `filename`.
    '''    
    # create melspectrogram using waveform and sampling rate
    S = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
    # calculate decibels
    S_dB = librosa.power_to_db(S)

    # create plot
    fig, ax = plt.subplots()
    img = ld.specshow(S_dB, sr=sampling_rate, ax=ax)

    # save images as .png in imgs directory
    plt.savefig(
        'imgs/' + filename[:-4] + modifier + '.png',
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.close()
