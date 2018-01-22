import random
import glob
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

from image_cropper import crop


class Pyramid:

    def __init__(self, model, threshold=0.99):
        self.model = model
        self.threshold = threshold
        self.chunk = 36

    def __model_wrapper(self, img):

        cpy_img = np.asarray(img)
        cpy_img = np.expand_dims(cpy_img, axis=0)
        cpy_img = np.expand_dims(cpy_img, axis=3)

        return self.model.predict(cpy_img)[0, 1] > self.threshold

    @staticmethod
    def __sample_img(img, strides, chunk):
        """
        Generator. sample an image given the parameters
        param img : the image to sample
        param strides : tuple (sx, sy); step between two analyzed chunk
        param chunk : integer, size of a square returned
        return : x unstrided, y unstrided, and a sample of img of shape (chunk, chunk, 3), padding valid
        """
        for x in range(0, img.shape[0], strides[0]):
            if x + chunk > img.shape[0]:
                break
            for y in range(0, img.shape[1], strides[1]):
                if y + chunk > img.shape[1]:
                    break
                yield x // strides[0], y // strides[0], img[x:x + chunk, y:y + chunk]

    def apply_model(self, img, scale, strides, padding, model):
        """
        analyze a whole image using a unique model (similar working than upgraded convolution)
        param img : PIL image representing the whole image to be analyzed
        param scale : float, in range [0;1], scaling applied to the image before analyzing
        param strides : tuple (sx, sy); step between two analyzed chunk
        param padding : doesn't affect analyzing right now
        param model : how to analyze the image. Input must be 36x36, output must be integer
        return: a list of nparrays of size (imgL*scale/stridex, imgl*scale/stridey) rounded to the floor,
                where each scale is decreased by 20% each time (if you begin with scale 1, you have :
                [outs1, outs0.8, outs0.64, ...])
        """

        if int(img.size[0] * scale) < self.chunk or int(img.size[1] * scale) < self.chunk:
            return []
        img_cpy = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        img_as_arr = np.asarray(img_cpy)
        output = - np.ones((img_as_arr.shape[0] // strides[0], img_as_arr.shape[1] // strides[1]))
        sampler = self.__sample_img(img_as_arr, strides, self.chunk)
        x = 0
        y = 0
        for x, y, sample in sampler:
            output[x, y] = self.__model_wrapper(sample)

        output = [(output[:x, :y], scale)]
        output += self.apply_model(img, scale * 0.8, strides, padding, model)
        return output

    def draw_on_image(self, img, matched, strides, scale):
        cpy = img.copy()
        cpy = cpy.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        draw = ImageDraw.Draw(cpy)
        for y, line in enumerate(matched):
            for x, value in enumerate(line):
                if value == 1:
                    draw.rectangle([x * strides[0], y * strides[1],
                                    x * strides[0] + self.chunk, y * strides[1] + self.chunk],
                                   fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128))
        cpy = cpy.resize(img.size)

        return cpy

    def get_visage_location(self, filename, strides=(9, 9)):
        test_img = Image.open(filename).convert('L')

        out = self.apply_model(test_img, 0.25, strides, None, None)

        return out

    def add_false_positive_to_negative_db(self, directory_path,
                                          save_path="/home/francois/Documents/tmp/test_image_crop/IMG_C{}_{}_{}.png",
                                          strides=(9, 9)):
        number_of_analyzed = 0
        passing = True
        for k, path in enumerate(glob.iglob("{}/*".format(directory_path))):
            print(path)
            try:
                visage_location = self.get_visage_location(path, strides)

                image_to_crop = Image.open(path)
                for labels_grid, scale in visage_location:
                    for j, line in enumerate(labels_grid):
                        for i, label in enumerate(line):
                            number_of_analyzed += 1
                            if label == 1:
                                crop(save_path.format(k, i, j),
                                     image_to_crop, (i, j),
                                     scale,
                                     strides,
                                     self.chunk)
            except:
                print('Unable to load image {}, doesn\'t matter, skip'.format(path))

        print("number of analyzed : ", number_of_analyzed)

    def test_pyramide(self, filename, strides=(9, 9)):

        test_img = Image.open(filename).convert('L')

        out = self.apply_model(test_img, 1, strides, None, None)
        t2 = test_img.copy().convert('RGBA')

        blank = Image.new(mode='RGBA', size=t2.size, color=(0, 0, 0, 0))
        for o, s in out[::-1]:
            blank = self.draw_on_image(blank, o, (9, 9), s)
        t2.paste(blank, (0, 0), mask=blank)

        plt.figure(figsize=(40, 40))
        plt.imshow(np.asarray(t2))
        plt.show()

