import os
import random
import glob
import shutil

import numpy as np
import cv2


from Pyramid import Pyramid

def import_model():
    from model import CustomModel
    return CustomModel


ROWS = 36
COLS = 36

data_directory = "start_deep/"
train_directory = os.path.join(data_directory, "train_images")
positives_text_file_path = os.path.join(data_directory, "positives.txt")
negatives_hard_directory = os.path.join(train_directory, "0_added")
negatives_hard_path = os.path.join(negatives_hard_directory, "*", "*")
negatives_path = os.path.join(train_directory, "0", "*")
added_images = os.path.join(data_directory, "added_images")


def get_negatives(negative_number):
    counter = 0
    for path in glob.iglob(negatives_hard_path):
        # print("acquisition::get_negatives : hard path:", path)
        yield path, 0
        counter += 1

    for path in glob.iglob(negatives_path):
        if counter >= negative_number:
            return
        # print("acquisition::get_negatives : easy path:", path)
        yield path, 0
        counter += 1

    return


def shuffle_and_cut(set_, labels, number):
    indices_shuffle = np.asarray(list(range(len(set_))))
    random.shuffle(indices_shuffle)

    set_rnd = np.asarray(set_)[indices_shuffle]
    labels_rnd = np.asarray(labels)[indices_shuffle]

    set_rnd = set_rnd[:number]
    labels_rnd = labels_rnd[:number]

    return set_rnd, labels_rnd


def get_trainset_list(positives_number=1000, negatives_number=1000):
    negatives_gen = get_negatives(negatives_number)
    negatives = list(negatives_gen)
    negative_images, negative_labels = zip(*negatives)
    negative_images_rnd, negative_labels_rnd = shuffle_and_cut(negative_images, negative_labels, negatives_number)

    negatives_number = len(negatives)
    positives_number = max(positives_number, negatives_number)

    with open(positives_text_file_path) as pf:
        positives = pf.readlines()
    positives = list(map(lambda x: x.strip().split(), positives))
    positive_images, positive_labels = zip(*positives)
    positive_images_rnd, positive_labels_rnd = shuffle_and_cut(positive_images, positive_labels, positives_number)
    positive_images_rnd = list(map(lambda x: x if train_directory in x else os.path.join(train_directory, x),
                                   positive_images_rnd))

    print("acquisition::get_trainset_list : positive_sample : {}".format(positive_images_rnd[:10]))
    print("acquisition::get_trainset_list : negative_sample : {}".format(negative_images_rnd[:10]))
    return positive_images_rnd, positive_labels_rnd, negative_images_rnd, negative_labels_rnd


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  #
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        try:
            image = read_image(image_file)
            data[i] = image.T
        except AttributeError:
            print("train_directory :", train_directory)
            print("image_file :", image_file)
            raise
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))

    return data


def save_hard_images(threshold_list=(0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5), image_number=1250):

    CustomModel = import_model()
    model = CustomModel()
    factor = 1
    for i, threshold in enumerate(threshold_list):
        print("acquisition::get_hard_images : threshold :", threshold)
        try:

            shutil.rmtree(os.path.join(negatives_hard_directory, str(threshold)))
        except FileNotFoundError:
            pass
        os.makedirs(os.path.join(negatives_hard_directory, str(threshold)))

        model.train(image_number, image_number)
        pyramid = Pyramid(model, threshold)
        pyramid.add_false_positive_to_negative_db(directory_path=added_images,
                                                  save_path=os.path.join(negatives_hard_directory,
                                                                         str(threshold)),
                                                  strides=(9, 9))

        pyramid = Pyramid(model, threshold_list[max(0, i-1)])
        pyramid.test_pyramide('photo_famille.jpeg')

        factor *= 0.8
        model.recompile(factor)


