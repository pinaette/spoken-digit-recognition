# Spoken Digit Recognition

Speech recognition for digits (0-9) in English, implemented as the final project for the Data Science Bootcamp at Lighthouse Labs.

What this project does:
- Create mel spectrograms from audio files.
- Generate data by adding noise, increasing amplitude, changing pitch and speed.
- Train a convolutional neural network using mel spectrograms of both original and generated data.

Many approaches found online that utilise the power of the mel scale do this by using coeffficients (mel-frequency cepstral coefficients, MFCCs for short). This is definitely much less computationally expensive, but this project uses the spectrograms themselves as image files since this is what CNNs are best known for: image processing. Using coefficients likely better lends itself to the use of LSTMs instead, for example.

Without the generated data, the original 3000 datapoints easily reaches 98% validation accuracy within 50 epochs. The entire dataset of 12000 datapoints, on the other hand, was able to reach 99% within the same timeframe, with 99.5% in around 85 epochs. This method of generating data was implemented because while 3000 datapoints is a good number, the original dataset is comprised of only 6 different males. Please note that both original and noise data were indiscriminately used as part of training and test datasets, in order to approximate real life conditions where speakers might have background noise etc.

This commit includes only sample data (two files per digit) as I didn't want to reupload the entire dataset. The full dataset used can be found here: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset). You can simply add this in a `data/` folder and comment/uncomment the relevant line (`filepath = ...`) in `final.py` before running it.

### What else?
This project aims to be a proof of concept only, with the entire process from audio file to trained model included. For real-world applications, steps such as below (and more) would be needed:
- Handling multi-digit inputs, such as phone numbers.
- Online deployment that takes audio input from user and predicts the digit.
- More varied dataset to include different voices, pitches, accents.
- Stretch: different languages.
- Stretch: more numbers, as opposed to single digits, e.g. "hundred" and "one" would yield "101" instead of "100" and "1".

## How to use this repository

This was designed to run in-folder as all paths are relative and most of the code doesn't take filepath as input. To reproduce in its entirety, you can download the original dataset from the link provided above and place them in `data/`, then change the `filepath` variable as explained.

Since it's best to run this on a GPU, here are the steps to reproduce this in Google Colab (which is what I did):
- Clone this folder into Google Drive.
- Mount Google Drive on Google Colab.
- Run this in your Colab notebook with runtime set to GPU:
``` python
import os
os.chdir('drive/MyDrive/spoken-digit-recognition')
!pip install -q -U tensorflow_addons
!python final.py
```

## Resources
- [Sound Augmentation Librosa (Kaggle)](https://www.kaggle.com/code/huseinzol05/sound-augmentation-librosa/notebook)
- [_Spoken Digit Recognition (Speech Recognition)_ by Avi Khemani](http://cs230.stanford.edu/projects_fall_2020/reports/55617928.pdf)
- CNN architecture adapted from: [Voice Classification: Urban_Sounds_Challenge by jurgenarias](https://github.com/jurgenarias/Portfolio/blob/master/Voice%20Classification/Code/Urban_Sounds_Challenge/Urban_Sounds_Classifier_CNN.ipynb)
- On working with the `image_dataset_from_directory` function:
    - [Keras API: Image data preprocessing](https://keras.io/api/preprocessing/image/)
    - [Keras code examples: Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- On working with Cyclical Learning Rate:
    - Original article: [_Cyclical Learning Rates for Training Neural Networks_ by Leslie N. Smith](https://arxiv.org/abs/1506.01186)
    - [Tensorflow add-on documentation](https://www.tensorflow.org/addons/tutorials/optimizers_cyclicallearningrate)
