import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

from NaiveClassifier import NaiveClassifier
from FeatureExtractor import FeatureExtractor
from FeatureExtractor import ColourHistogramExtractor,HoGExtractor
from ClassificationMetrices import ClassificationMetrices


""" 
main driver file
"""

def main(file_name):

	#cifar 10 data 
	def unpickle(file):
	    with open(file, 'rb') as fo:
	        data = pickle.load(fo, encoding ='latin1')
	    return data

	def data_process():
		# data_set
		data_set = unpickle(file_name)

		labels = data_set['labels']
		data = data_set['data']
		filenames=data_set['filenames']

		#number of images, number of channels, x dim, y dim
		data = np.reshape(data, (10000, 3, 32, 32))
		#print(data.shape)
		train_img = data[:10]
		test_img = data[10:20]

		#training label
		train_label = labels[:10]
		test_label = labels[10:20]

		return train_img, train_label, test_img, test_label


	train_img, train_label, test_img, test_label = data_process()

	#extractor object init
	feature1 = ColourHistogramExtractor
	#feature1 = HoGExtractor

	#naive classifier
	img_class = NaiveClassifier( train_img, train_label, feature1)
	#feature_extracted = img_class.extract_feature_from_single_image(train_img[4])

	#single images
	print("For single images")
	predicted_label,predicted_score = img_class.classify_single_image(test_img[0])
	print("Class: ",predicted_label)
	print("Score: ",predicted_score)

	#multiple images
	print("\nFor multiple images")
	predicted_label, predicted_score = img_class.classify_multiple_images(test_img)

	for x in range(0,len(predicted_score)):
		print("Prediction Label: ",predicted_label[x])
		print("Score: ",predicted_score[x])

	#finding accuracy , precision, recall , harmonic mean
	classify_class = ClassificationMetrices(test_label,predicted_label,predicted_score)
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

	while(1):
		if len(sys.argv) == 2:
			main(sys.argv[1])
			exit(0)
		else:
			print("Dataset is not found!\n")
			print("Try Again\n")





