import os
import time
from processing import *

# create relative directory within current directory to store images
os.mkdir('imgs')
# use relative directory to get list of filenames as string
filepath = 'data/'
# filepath = 'sample_data/'

files = os.listdir(filepath)

# get time before starting image creation
print('creating mel spectrograms...')
print('...')
tick = time.time()
# store mel spectrograms as images
for file in files:
    plot_mel_from_wav(path=filepath, filename=file)
# get time after image creation is done
duration = time.time() - tick

# display elapsed time
print('spectrograms created')
print('elapsed time:', human_readable_time(duration))
print('...')

# set category labels for digits
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# get list of img file names as string
imgs = os.listdir('imgs/')

# create subdirectories for each label
for label in labels:
    os.mkdir('imgs/'+label)

# move each image to its subdirectory
print('moving spectrograms to subfolders per digit')

for img in imgs:
    lbl = img[0]
    os.rename('imgs/'+img, 'imgs/'+lbl+'/'+img)

print('all spectrograms labelled!')
