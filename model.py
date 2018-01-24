import random

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Layer
from keras.models import Model
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.losses import cosine_proximity, categorical_crossentropy, mean_squared_error
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras import backend as K

import numpy as np

from scipy.spatial.distance import cdist

from acquisition import get_trainset_list, prep_data


# Custom loss layer
class CustomDistanceLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomDistanceLossLayer, self).__init__(**kwargs)

    def distance_loss(self, anchor, positive, negative):
        l2_anchor = K.l2_normalize(anchor, axis=1)
        l2_positive = K.l2_normalize(positive, axis=1)
        l2_negative = K.l2_normalize(negative, axis=1)
        loss = cosine_proximity(l2_anchor, l2_positive) - cosine_proximity(l2_anchor, l2_negative) # environ 10 max next time increase factor value
        return K.mean(loss, axis=-1) / 10

    def call(self, inputs):
        x = inputs[0]
        positive = inputs[1]
        negative = inputs[2]
        loss = self.distance_loss(x, positive, negative)
        self.add_loss(loss, inputs=inputs)
        return x


# Custom loss layer
class CustomFinalLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomFinalLossLayer, self).__init__(**kwargs)

    def final_loss(self, image, label, image_decoded, label_predicted):
        # environ 15 max
        loss = (K.mean(categorical_crossentropy(label, label_predicted), axis=-1) / 15) \
             + (K.mean(mean_squared_error(image, image_decoded), axis=-1) / 20000) # environ 20000 max
        return loss

    def call(self, inputs):
        image = inputs[0]
        label = inputs[1]
        image_decoded = inputs[2]
        label_predicted = inputs[3]
        loss = self.final_loss(image, label, image_decoded, label_predicted)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return label_predicted


class CustomModel:
    def __init__(self, filename=None):
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

        self.encoder = Model(input_layer, to_be_densed, name='encoder')

        anchor = Input(shape=(36, 36, 1))
        positive = Input(shape=(36, 36, 1))
        negative = Input(shape=(36, 36, 1))
        labels = Input(shape=(2,))

        cosine_loss = CustomDistanceLossLayer()([self.encoder(anchor), self.encoder(positive), self.encoder(negative)])

        upsample = UpSampling2D()

        conv_3_func = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        reshaping = Reshape((9, 9, 2))
        conv_4_func = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        conv_5_func = Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding="same", activation="relu")

        input_decoder = Input((162,))
        reshaped = reshaping(input_decoder)
        conv_3 = conv_3_func(reshaped)
        up_1 = upsample(conv_3)
        conv_4 = conv_4_func(up_1)
        up_2 = upsample(conv_4)
        output_decoder = conv_5_func(up_2)

        self.decoder = Model(input_decoder, output_decoder, name='decoder')

        autoencoder_output = self.decoder(cosine_loss)

        output_categorical_layer = fully_connected(cosine_loss)

        output = CustomFinalLossLayer()([anchor, labels, autoencoder_output, output_categorical_layer])
        self.model_to_train = Model([anchor, positive, negative, labels], [output], name='full network')
        input_value = Input(shape=(36, 36, 1))
        output_value = fully_connected(self.encoder(input_value))
        self.model_to_predict = Model(input_value, output_value)

        if filename is not None:
            self.model_to_train.load_weights(filename)
        self.model_to_train.compile('adam', None, ['accuracy'])

        self.epochs = 15

    def select_triplets(self, anchors, labels):

        if len(anchors)%50 != 0:
            anchors = anchors[:-(len(anchors)%50)]
            labels = labels[:-(len(labels)%50)]
        anchors_embeddings = self.encoder.predict(anchors)
        distances = cdist(anchors_embeddings, anchors_embeddings, metric='cosine')
        positives_labels = []
        negatives_labels = []
        for i, line in enumerate(distances):
            positives_distances = line.copy()
            positives_distances[np.where(labels != labels[i])] *= 0
            max_positive = positives_distances.max()
            positives_labels.append(np.argmax(positives_distances))
            negative_distances = line.copy()
            if (max_positive != line.max()):
                negative_distances[np.where(negative_distances <= max_positive)] *= 5000
            negative_distances[np.where(labels == labels[i])] *= 5000
            negative_distances[i] = 5000000
            negatives_labels.append(np.argmin(negative_distances))

        return [anchors, anchors.copy()[positives_labels], anchors.copy()[negatives_labels], to_categorical(labels)]

    def train(self, positive_images=700, negative_images=700):
        anchors_positive, positive_labels, anchors_negative, negative_labels = get_trainset_list(positive_images,
                                                                                                 negative_images)

        anchors = np.concatenate((anchors_positive, anchors_negative), axis=0)

        anchors_labels = np.concatenate((positive_labels, negative_labels))

        indices = list(range(len(anchors)))

        random.shuffle(indices)

        anchors = np.expand_dims(prep_data(anchors[indices]), axis=3)
        anchors_labels = np.asarray(list(map(lambda x: int(x), anchors_labels[indices])))

        for i in range(self.epochs):
            print("epochs ", i)
            self.model_to_train.fit(x=self.select_triplets(anchors, anchors_labels), batch_size=50)

    def predict(self, image):
        return self.model_to_predict.predict(image)

    def recompile(self, factor):
        self.model_to_train.compile(optimizers.Adam(lr=0.001*factor), None, ['accuracy'])
        #self.epochs /= factor
        #self.epochs = int(self.epochs)
