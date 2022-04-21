from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import CyclicalLearningRate

def get_data(labels, subset, path='imgs/', batch_size=128, image_size=(187, 250), seed=126):
    '''
    uses a keras function to automatically create tf.dataset from image files.
    takes input as:
        - path to image files directory. requires the files to have been separated into classes/labels beforehand.
        - output classes as labels.
        - batch_size for training.
        - image_size can be changed depending on the dataset. 187x250 was big enough while still being somewhat efficient.
        - this should be used to generate train and test datasets separately, specified as "subset" at time of running.
        - seed is mandatory when using validation_split to ensure train and test datasets don't overlap.
    return created dataset.
    '''
    dataset = image_dataset_from_directory(
        directory = path,
        labels = 'inferred',
        label_mode = 'categorical',
        class_names = labels,
        image_size = image_size,
        validation_split = 0.15,
        subset = subset,
        crop_to_aspect_ratio = True,
        seed = seed
    )

    return dataset

def compile_model():

    # initialize optimizer parts
    learning_rate = CyclicalLearningRate(
        initial_learning_rate = 0.0005,
        maximal_learning_rate = 0.005,
        step_size = 160,
        scale_fn = lambda x: 1/(2.**(x-1)) # lambda function for triangular2 method
    )
    optimizer = Adam(learning_rate=learning_rate)

    # initialize model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(187, 250, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['accuracy']
    )
    
    return model
