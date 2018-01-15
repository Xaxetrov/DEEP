from PIL import Image
import os


def crop(save_path, path_to_image, height, width):
    k = 0
    im = Image.open(path_to_image)
    img_width, img_height = im.size
    for i in range(0, img_height, height):
        for j in range(0, img_width, width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            # try:
            #o = a.crop(area)
            a.save(os.path.join(save_path, "{}".format(page), "IMG-{}.png".format(k)))
            # except:
            #     pass
            k += 1
