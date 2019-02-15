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
imgs = []

for i in range(len(arr)):
    temp = plt.imread("./data/"+arr[i]+".png")
    imgs.append(temp)
imgs = np.array( imgs )


#number of images, number of channels, x dim, y dim
train_img = imgs.transpose(0,3,1,2)
imgs = train_img
#print(imgs[0].shape)
#print(trainImg.shape)


#training label
train_label = np.array([0,0,0,1,1])

#object init
feature1 = ColourHistogramExtractor

img_class = NaiveClassifier( train_img, train_label, feature1 )
feature_extracted = img_class.extract_feature_from_single_image( imgs[0] )


