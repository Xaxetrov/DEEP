import random

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras import optimizers

import numpy as np

from acquisition import get_trainset_list, prep_data


class CustomModel:
    def __init__(self):
        pool_func = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same")

        conv_1_func = Conv2D(filters=4, kernel_size=(5, 5), strides=1, padding="same", activation="relu")
        conv_2_func = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        fully_connected = Dense(2, activation="softmax")

        input_layer = Input(shape=(36, 36, 1))
        conv_1 = conv_1_func(input_layer)
        pool_1 = pool_func(conv_1)
        conv_2 = conv_2_func(pool_1)
        pool_2 = pool_func(conv_2)
        to_be_densed = Flatten()(pool_2)
        output_categorical_layer = fully_connected(to_be_densed)

        self.model_to_train = Model(input_layer, output_categorical_layer)

        self.model_to_train.compile('adam', 'categorical_crossentropy', ['accuracy'])

        self.epochs = 25

    def train(self, positive_images=700, negative_images=700):

        positive_images, positive_labels, negative_images, negative_labels = get_trainset_list(positive_images,
                                                                                               negative_images)

        mischung_images = np.concatenate((positive_images, negative_images), axis=0)
        mischung_labels = np.concatenate((positive_labels, negative_labels))

        indices = list(range(len(mischung_images)))

        random.shuffle(indices)

        mischung_images = prep_data(mischung_images[indices])
        mischung_labels = mischung_labels[indices]
        mischung_images = np.expand_dims(mischung_images, axis=3)

        self.model_to_train.fit(x=mischung_images, y=to_categorical(mischung_labels), batch_size=32, epochs=self.epochs,
                                validation_split=0.1, shuffle=True)

    def predict(self, image):
        return self.model_to_train.predict(image)

    def recompile(self, factor):
        self.model_to_train.compile(optimizers.Adam(lr=0.001*factor), 'categorical_crossentropy', ['accuracy'])
        self.epochs /= factor
        self.epochs = int(self.epochs)
