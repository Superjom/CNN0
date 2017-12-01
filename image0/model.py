from __future__ import  division
import tensorflow as tf
from load_data import load_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.utils import np_utils
import numpy as np

img_width, img_height = 192, 108

datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        )

train_data_dir = '../people.1.txt'
test_data_dir = '../people.test.txt'
prediction_out = './prediction.txt'

batch_size = 30

X_train, Y_train = load_dataset(train_data_dir)
X_test, Y_test = load_dataset(test_data_dir)
X_valid, Y_valid = load_dataset('../people.valid.txt')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_valid -= mean_image

X_train /= 128.
X_test /= 128.
X_valid /= 128.

nb_classes = 2
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)



from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

shape = (img_height, img_width, 3)
print 'shape', shape

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

for i in range(0):
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

nb_epoch=30

class ValidPrediction(Callback):
    counter = 0
    best_score = 0.
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.counter % 20 == 0:
            res = model.predict(X_valid, verbose=1)
            labels = res[:, 1]
            #labels = np.array([1 if l > 0.6 else 0 for l in labels], dtype='float32')
            labels1 = np.array([1. if l > 0.8 else 0 for l in labels], dtype='float32')
            print Y_valid[:,1][:10]
            print labels1[:10]
            print ' '.join('{:1.1f}'.format(f) for f in np.abs(Y_valid[:,1] - labels)[:10])
            print ' '.join('{:1.1f}'.format(f) for f in np.abs(Y_valid[:,1] - labels1)[:10])
            error = np.mean((labels - Y_valid[:,1])**2) 
            error1 = np.mean((labels1 - Y_valid[:,1])**2)
            print 'pos', 1-error
            print 'pos1', 1-error1

            if 1-error1 > self.best_score:
                with open(prediction_out, 'w') as f:
                    f.write('\n'.join(map(str, labels1)))
                self.best_score = 1-error1
            
        self.counter += 1

valid_prediction = ValidPrediction()


model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    validation_data=(X_test, Y_test),
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[
                        valid_prediction,
                        ])
