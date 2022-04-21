print('importing libraries...')

import os
import time

from processing import *
from modelling import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print('setting variables...')
# get time before the entire process
start_time = time.time()

# create relative directory within current directory to store images
os.mkdir('imgs')
# use relative directory to get list of filenames as string
filepath = 'data/'
# filepath = 'sample_data/'
files = os.listdir(filepath)

# initialize dictionary for storing each datapoint and its augmented versions
samples = dict()

print('creating mel spectrograms...')
print('...')

# get time before starting image creation
tick = time.time()

# store mel spectrograms as images
for file in files:
    # get waveform and sampling rate
    samples['_original'], sr = load_audio(path=filepath, filename=file)
    # get augmented data
    samples['_pitchspeed'], samples['_amplified'], samples['_noise'] = oversample(samples['_original'])
    for k, v in samples.items():
        plot_mel_from_waveform(waveform=v, sampling_rate=sr, filename=file, modifier=k)

# display elapsed time for image creation
duration = time.time() - tick
print('spectrograms created')
print('elapsed time for creating spectrograms:', human_readable_time(duration))
print('...')

# get list of img file names as string
imgs = os.listdir('imgs/')

# create subdirectories for each label
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for label in labels:
    os.mkdir('imgs/'+label)

# move each image to its respective subdirectory
print('moving spectrograms to subfolders per digit')
for img in imgs:
    lbl = img[0]
    os.rename('imgs/'+img, 'imgs/'+lbl+'/'+img)

print('all spectrograms labelled!')
print('...')
print('creating training and test datasets...')
tick = time.time()

# create datasets
train = get_data(labels=labels, subset='training')
test = get_data(labels=labels, subset='validation')

# display elapsed time for dataset creation
duration = time.time() - tick
print('datasets created')
print('elapsed time for creating datasets:', human_readable_time(duration))
print('...')

print('compiling model...')
model = compile_model()

# set callbacks
cp = 'checkpoint/'
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10),
    ModelCheckpoint(
        filepath = cp,
        monitor = 'val_accuracy',
        save_best_only = True,
        save_weights_only = True,
        mode='max')
]

tick = time.time()
model.fit(
    train,
    epochs = 100,
    verbose = 1,
    callbacks = callbacks,
    validation_data = test
)
# display elapsed time for training
duration = time.time() - tick
print('training complete')
print('elapsed time for training:', human_readable_time(duration))
print('...')

# display total time running this code
total_time = time.time() - start_time
print('total time spent:', human_readable_time(total_time))
