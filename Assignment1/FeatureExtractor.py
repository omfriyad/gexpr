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

    def extract_feature(self):
        return self.hog()

    def gradient(self):
        """THis will return a same size of image gradient for HOG"""
        gx = np.zeros(self.img.shape)
        gx[:, 1:-1] = -self.img[:, :-2] + self.img[:, 2:]
        gx[:, 0] = -self.img[:, 0] + self.img[:, 1]
        gx[:, -1] = -self.img[:, -2] + self.img[:, -1]

        gy = np.zeros(self.img.shape)
        gy[1:-1, :] = self.img[:-2, :] - self.img[2:, :]
        gy[0, :] = self.img[0, :] - self.img[1, :]
        gy[-1, :] = self.img[-2, :] - self.img[-1, :]

        return gx, gy

    def magnitude(self):
        gx , gy = self.gradient()

        return np.sqrt(gx**2, gy**2)

    def orientation(self):
        gx , gy = self.gradient()

        return np.rad2deg(arctan2(gx, gy))%180 #returning in 180 degree

    def hog(self):
        pass

