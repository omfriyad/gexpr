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


#object init
f1 = ColourHistogramExtractor(imgD[0])
colorhistogram = f1.colorHis()
print(colorhistogram.shape)

