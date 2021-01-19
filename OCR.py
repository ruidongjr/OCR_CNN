import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2


def LeNet5(input_shape=(32, 32, 1), classes=10):
    """
    Implementation of a modified LeNet-5.
    Modified Architecture -- ConvNet --> Pool --> ConvNet --> Pool --> (Flatten) --> FullyConnected --> FullyConnected --> Softmax

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential([

        # Layer 1
        Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 1), name='convolution_1'),
        MaxPooling2D(pool_size=2, strides=2, name='max_pool_1'),

        # Layer 2
        Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', name='convolution_2'),
        MaxPooling2D(pool_size=2, strides=2, name='max_pool_2'),

        # Layer 3
        Flatten(name='flatten'),
        Dense(units=120, activation='relu', name='fully_connected_1'),

        # Layer 4
        Dense(units=84, activation='relu', name='fully_connected_2'),

        # Output
        Dense(units=10, activation='softmax', name='output')

    ])

    model._name = 'LeNet5'

    return model

def preprocess():

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    Y = train[['label']]
    X = train.drop(train.columns[[0]], axis=1)
    X = X.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
    print("Size of Dataset: " , len(X))
    cross_validation_size = int(len(X)*0.05)
    print("Size of Cross Validation Set: " , cross_validation_size)
    random_seed = 2
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = cross_validation_size, random_state=random_seed)
    X_test = test

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    # Padding the images by 2 pixels since in the paper input images were 32x32
    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    # Standardization
    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)
    X_train = (X_train - mean_px)/(std_px)
    mean_px = X_val.mean().astype(np.float32)
    std_px = X_val.std().astype(np.float32)
    X_val = (X_val - mean_px)/(std_px)
    mean_px = X_test.mean().astype(np.float32)
    std_px = X_test.std().astype(np.float32)
    X_test = (X_test - mean_px)/(std_px)
    # One-hot encoding the labels
    Y_train = to_categorical(Y_train, num_classes = 10)
    Y_val = to_categorical(Y_val, num_classes = 10)

    return X_train, X_val, Y_train, Y_val, X_test

def train(X_train, X_val, Y_train, Y_val):
    LeNet5Model = LeNet5(input_shape = (32, 32, 1), classes = 10)
    LeNet5Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    LeNet5Model.summary()

    datagen = ImageDataGenerator(
            featurewise_center = False,  # set input mean to 0 over the dataset
            samplewise_center = False,  # set each sample mean to 0
            featurewise_std_normalization = False,  # divide inputs by std of the dataset
            samplewise_std_normalization = False,  # divide each input by its std
            zca_whitening = False,  # apply ZCA whitening
            rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip = False,  # randomly flip images
            vertical_flip = False)  # randomly flip images
    datagen.fit(X_train)

    variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

    history = LeNet5Model.fit(X_train, Y_train, epochs = 30, batch_size = 64, callbacks = [variable_learning_rate], validation_data = (X_val,Y_val))

    # save model
    model_name = 'model/OCR'
    LeNet5Model.save(model_name)

if __name__ == "__main__":
    # X_train, X_val, Y_train, Y_val, X_test = preprocess()
    l = [0, 1, 2, 3]
    for i in l:
        image = "data/image_{}.jpg".format(i)
        image = cv2.imread(image)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        ht, wd, cc = image.shape
        ww = 32
        hh = 32
        color = (0, 0, 0)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd] = image
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img = img[np.newaxis, :, :, np.newaxis]
        # X_test = np.pad(image, ((0,0),(2,2),(2,2),(0,0)), 'constant')

        model_name = 'model/OCR'
        ocr_model = load_model(model_name)
        results = ocr_model.predict(img)
        results = np.argmax(results, axis = 1)
        print(results)