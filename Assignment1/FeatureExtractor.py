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

        r,g,b = self.img[0],self.img[1],self.img[2]

        freR = self.fre(r)
        freG = self.fre(g)
        freB = self.fre(b)

        freFinal = np.maximum.reduce([freR,freG,freB])

        # plt.plot(freR,color="Red")
        # plt.plot(freG,color="Green")
        # plt.plot(freB,color="Blue")
        # plt.plot(freFinal,color="Black")

        # plt.ylabel('Frequency')
        # plt.show()

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

    def direction(self):
        gx , gy = self.gradient()

        return np.rad2deg(arctan2(gx, gy))%180 #returning in 180 degree
    @staticmethod
    def interpolation(magnitude, degree):

        idx = degree // 20  # ie 17 is the degree it fells between index 0-1 or 0 - 20 and bin size is 9
                            # 17//20 = 0 so first index = 0 2nd one is 1
        percentage = degree / 20

        return (idx%9, magnitude*(1-percentage)), ((idx+1)%9, magnitude*percentage)
    def block_histogram(self,magnitude_block, degree_block):
        hist = np.zeros((9,))
        for x in range(8):
            for y in range(8):
                a ,b = self.interpolation(magnitude_block[x,y], degree_block[x,y])
                hist[a[0]] = hist[a[0]] + a[1]
                hist[b[0]] = hist[b[0]] + b[1]

        return hist

    def generate_per_block_histogram(self):
        hist_vector = np.zeros((8,16))
        magnitude = self.magnitude()
        degree = self.direction()

        for y in range(0, 128, 8):
            for x in range(0, 64, 8):
                mag_block = magnitude[x:x+7, y:y+7]
                degree_block  = degree[x:x+7, y:y+7]

                hist_vector[x//8,y//8] = self.block_histogram(mag_block, degree_block)

        return hist_vector # will return 8x16 vection from which we can work on 16x16 or 4x4 grid



    def hog(self):
        return self.magnitude() # dummy

