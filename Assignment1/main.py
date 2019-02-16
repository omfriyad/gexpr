import numpy as np
from matplotlib import pyplot as plt

from NaiveClassifier import NaiveClassifier
from FeatureExtractor import FeatureExtractor
from FeatureExtractor import ColourHistogramExtractor
from ClassificationMetrices import ClassificationMetrices


""" 
main driver file
"""

def main():

	# data 
	arr = [ "horse1","horse2","horse3",
	            "cat1","cat2"  ]
	class_id =["horse","cat"]
	imgs = []

	for i in range(len(arr)):
	    temp = plt.imread("./data/"+arr[i]+".png")
	    imgs.append(temp)
	imgs = np.array( imgs )


	#number of images, number of channels, x dim, y dim
	train_img = imgs.transpose(0,3,1,2)
	imgs = train_img


	#training label
	train_label = np.array([0,0,0,1,1])


	#extractor object init
	feature1 = ColourHistogramExtractor

	#naive classifier
	img_class = NaiveClassifier( train_img, train_label, feature1)
	#feature_extracted = img_class.extract_feature_from_single_image( imgs[0] )

	#single images
	print("For single images")
	test_img = imgs[1]
	prediction_label, score = img_class.classify_single_image(test_img)
	print("Class: ",prediction_label," -> ",class_id[prediction_label])
	#print(prediction_label)
	print("Score: ",score[0])


	#multiple images
	print("\nFor multiple images")
	test_img = imgs[2:5]
	prediction_label, score = img_class.classify_multiple_images(test_img)

	for x in range(0,len(score)):
		print("Class: ",prediction_label[x]," -> ",class_id[prediction_label[x]])
		print("Score: ",score[x])


def main2():

	# y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
	# y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
	# y_score = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

	y_true = [ 1, 0, 1, 1, 0 ]
	y_pred = [ 0, 0, 1, 1, 0 ]
	y_score =[ 0, 0, 1, 1, 0 ]

	classify_class = ClassificationMetrices(y_true,y_pred,y_score)
	#a = classify_class.get_confusion_matrix_for_heatmap()
	#a = classify_class.calculate_accuracy()
	a = classify_class.calculate_precision()
	b = classify_class.calculate_recall()
	print(a," || ",b)


if __name__  == "__main__":
	main2()







