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


	def data_process():
		# data 
		data_set = [ "horse1","horse2","horse3",
		            "cat1","cat2"  ]
		imgs = []

		for i in range(len(data_set)):
		    temp = plt.imread("./data/"+data_set[i]+".png")
		    imgs.append(temp)
		imgs = np.array( imgs )


		#number of images, number of channels, x dim, y dim
		imgs = imgs.transpose(0,3,1,2)
		train_img = imgs
		test_img = imgs[3]

		#training label
		train_label = np.array([0,0,0,1,1])
		test_label = np.array([0,1,1])

		return train_img, train_label, test_img, test_label


	#horse = 0 , cat = 1
	class_id =["horse","cat"]
	train_img, train_label, test_img, test_label = data_process()

	#extractor object init
	feature1 = ColourHistogramExtractor

	#naive classifier
	img_class = NaiveClassifier( train_img, train_label, feature1)
	#feature_extracted = img_class.extract_feature_from_single_image( imgs[0] )

	#single images
	# print("For single images")
	# predicted_label, score = img_class.classify_single_image(test_img)
	# print("Class: ",predicted_label," -> ",class_id[predicted_label[0]])
	# print("Score: ",score)

	#multiple images
	print("\nFor multiple images")
	test_img = train_img[2:5]
	predicted_label, predicted_score = img_class.classify_multiple_images(test_img)

	for x in range(0,len(predicted_score)):
		print("Class: ",predicted_label[x]," -> ",class_id[predicted_label[x]])
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

	main()





