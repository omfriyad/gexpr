import numpy as np
from numpy import arctan2
from matplotlib import pyplot as plt

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

    def __init__(self,img):
        super(ColourHistogramExtractor, self).__init__(img)
        #self.img = img

    def fre(self,color):
        fre = np.array([])
        for i in range(256):
            fre = np.append(fre, np.count_nonzero(color == i))
        total = np.sum(fre)
        fre = fre/total
        return fre


    def extract_feature(self):

        self.img = self.img*255
        self.img = self.img.astype(int)

        r,g,b = self.img[...,0],self.img[...,1],self.img[...,2]

        freR = self.fre(r)
        freG = self.fre(g)
        freB = self.fre(b)

        freFinal = np.maximum.reduce([freR,freG,freB])

        plt.plot(freR,color="Red")
        plt.plot(freG,color="Green")
        plt.plot(freB,color="Blue")
        plt.plot(freFinal,color="Black")

        plt.ylabel('Frequency')
        plt.show()

        return freFinal



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
        return self.magnitude() # dummy

