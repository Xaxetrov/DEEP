import random

import numpy as np
import cv2

ROWS = 36
COLS = 36

data_directory = "start_deep/"
train_directory = data_directory + "train_images/"
positives_path = data_directory + "positives.txt"
negatives_path = data_directory + "hard_neg_2_iter.txt"


def get_trainset_list(positives_number=1000, negatives_number=1000):
    with open(positives_path) as pf:
        positives = pf.readlines()
    positives = [x.strip() for x in positives]
    positives = list(map(lambda x: x.split(), positives))
    positive_images, positive_labels = zip(*positives)
    positive_shuffle = list(range(len(positive_images)))
    random.shuffle(positive_shuffle)
    positive_images_r = [positive_images[i] for i in positive_shuffle]
    positive_images_r = positive_images_r[:positives_number]
    positive_labels_r = [positive_labels[i] for i in positive_shuffle]
    positive_labels_r = positive_labels_r[:positives_number]

    with open(negatives_path) as nf:
        negatives = nf.readlines()
    negatives = [x.strip() for x in negatives]
    negatives = list(map(lambda x: x.split(), negatives))
    negative_images, negative_labels = zip(*negatives)
    negative_shuffle = list(range(len(negative_images)))
    random.shuffle(negative_shuffle)
    negative_images_r = [negative_images[i] for i in negative_shuffle]
    negative_images_r = negative_images_r[:negatives_number]
    negative_labels_r = [negative_labels[i] for i in negative_shuffle]
    negative_labels_r = negative_labels_r[:negatives_number]

    return positive_images_r, positive_labels_r, negative_images_r, negative_labels_r


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  #
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(train_directory + image_file)
        data[i] = image.T
        if i % 250 == 0 : print('Processed {} of {}'.format(i, count))

    return data

