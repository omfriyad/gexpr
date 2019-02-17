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

	y_true = [0,1,1,1,0]
	y_pred = [1,0,0,1,0]

	# y_true = [1,2,3,4,5,1,2,1,1,4,1]
	# y_pred = [1,2,3,4,5,1,2,1,1,4,5]

	y_score = [1,2,3,4,5,1,2,1,1,4,1]

	classify_class = ClassificationMetrices(y_true,y_pred,y_score)
	confusion_matrix = classify_class.get_confusion_matrix_for_heatmap()
	accuracy = classify_class.calculate_accuracy()
	precision = classify_class.calculate_precision()
	recall = classify_class.calculate_recall()
	f1 = classify_class.calculate_f1()

	print("Confusion Matrix\n", confusion_matrix)
	print("Accuracy: ", accuracy)
	print("Average Precision: ",precision)
	print("Average Recall: ",recall)
	print("Harmonic Mean: ",f1)

if __name__  == "__main__":
	main2()







