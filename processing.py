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

def plot_mel_from_wav(path, filename, sr=None):
    '''
    load audio as waveform, extract sampling rate, convert both into melspectrogram, calculate decibels, plot.

    waveform is a time series, represented as a 1d numpy floating point array.
    by default .load() samples at 22050 (number of samples per second of audio) and mixes to mono.
    when sampling_rate (sr) set to None, keeps the original sampling rate of the file (8kHz in this case)
    '''
    waveform, sampling_rate = librosa.load(path+filename, sr=sr)
    
    # create melspectrogram using waveform and sampling rate
    S = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
    # calculate decibels
    S_dB = librosa.power_to_db(S)

    # create plot
    fig, ax = plt.subplots()
    img = ld.specshow(S_dB, sr=sampling_rate, ax=ax)

    # save images as .png in imgs directory
    plt.savefig(
        'imgs/' + filename[:-3] + 'png',
        bbox_inches = 'tight',
        pad_inches = 0
        )
    plt.close()
