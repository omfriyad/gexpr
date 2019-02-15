import numpy as np
from matplotlib import pyplot as plt

from NaiveClassifier import NaiveClassifier
from FeatureExtractor import FeatureExtractor
from FeatureExtractor import ColourHistogramExtractor
from ClassificationMetrices import ClassificationMetrices


""" 
main driver file
"""

# data 
arr = [ "horse1","horse2","horse3",
            "cat1","cat2"  ]
imgD = []

for i in range(len(arr)):
    temp = plt.imread("./data/"+arr[i]+".png")
    imgD.append(temp)
imgD = np.array( imgD )


#number of images, number of channels, x dim, y dim
trainImg = imgD.transpose(0,3,1,2)
#print(trainImg.shape)


#training label
trainLabel = np.array([0,0,0,1,1])


#object init
feature1 = ColourHistogramExtractor
# colorhistogram = f1.colorHis()
# print(colorhistogram.shape)

imgClass = NaiveClassifier(trainImg,trainLabel,feature1)
i = imgClass.extract_feature_from_single_image(imgD[0])
# imgClass.h()


