import numpy as np
from numpy import arctan2


class FeatureExtractor(object):

    def __init__(self, img):
        self.img = img

    def extract_feature(self):
        return self.img.reshape(-1)



    """ 
    implement the following sub-classes of FeatureExtractor
        1. ColourHistogramExtractor
        2. HoGExtractor
        3. SIFTBoVWExtractor
    """

class ColourHistogramExtractor(FeatureExtractor):
    """docstring for ColourHistogramExtractor"""
    def __init__(self, img):
        super(FeatureExtractor, self).__init__()
        self.img = img

    def extract_feature(self):
        pass


class HoGExtractor(FeatureExtractor):
    """This is hog extractor """
    def __init__(self, img):
        super(HoGExtractor, self).__init__(img=img)
        self.img = img


    def gradient(self, image, same_size=False):
        sy, sx = image.shape
        if same_size:
            gx = np.zeros(image.shape)
            gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
            gx[:, 0] = -image[:, 0] + image[:, 1]
            gx[:, -1] = -image[:, -2] + image[:, -1]

            gy = np.zeros(image.shape)
            gy[1:-1, :] = image[:-2, :] - image[2:, :]
            gy[0, :] = image[0, :] - image[1, :]
            gy[-1, :] = image[-2, :] - image[-1, :]

        else:
            gx = np.zeros((sy - 2, sx - 2))
            gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

            gy = np.zeros((sy - 2, sx - 2))
            gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]

        return gx, gy


    def magnitude_orientation(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.rad2deg(arctan2(gy, gx)) % 180  #This operation results the into 180

        return magnitude, orientation

    def histogram_from_gradients(self, gx, gy, cell_size, cell_per_block, signed_orientation, nbins, visualize, normalise,
                                 flatten, same_size):


        pass



    def histogram_of_gradient(self):
        pass