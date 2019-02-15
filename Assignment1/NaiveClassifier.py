
import numpy as np

class NaiveClassifier:

	"""
	Create a class NaiveClassifier which accepts the following in 
	its constructor (init) in the following order
	1. Training images as a (number of images, number of channels, x dim, y dim) 
	   numpy array.For now assume every image in the training set should have 
	   the same number of channels and the same dimensions.
	2. Training labels. This should be a (number of images, 1) numpy array.
    3. An object of type FeatureExtractor
	"""
	def __init__(self, train_img, train_label, feature):
	    self.train_img = train_img
	    self.train_label = train_label
	    self.feature = feature

	""" 
	extract feature from single image, which should accept an image 
	as parameter. The image is a (number of channels, x dim, y dim) numpy array. 
	This method should return the feature extracted by calling the 
	extract feature method of the FeatureExtractor object.
	"""

	def extract_feature_from_single_image(self,img):
		self.feature = self.feature(img)
		return self.feature.extract_feature()


	"""
	extract feature from multiple images, which should accept a 
	(number of images, number of channels, x dim, y dim) numpy array 
	as parameter and return a (number of images, feature dim) array.
	"""
	def extract_feature_from_multiple_images(self):
		pass




	"""
	classify single image, which will accept an image as parameter 
	and return a label value and the corresponding score.
	"""
	def classify_single_image(self,img):


		def euclidean_distance(x, y):
		    distance = np.sum(np.square(x-y))
		    return np.sqrt(distance)
		
		distances = np.array([])
		neighbors = np.array([])
		label_index = np.array([])

		img = self.feature(img).extract_feature()

		for x in self.train_img:
			x = self.feature(x).extract_feature()
			dist = euclidean_distance(img,x)
			distances = np.append(distances, dist)

		prediction = np.argsort(distances)
		distances = np.sort(distances)

		# Naive 1NN Classifier , so k = 1
		k=1

		for x in range(k):
			neighbors = np.append(neighbors,distances[x])
			label_index = np.append(label_index,prediction[x])

		label_index = label_index.astype(int)
		label_value = self.train_label[label_index][0]

		return label_value,neighbors
		

	"""
	classify multiple images, which will accept multiple images as 
	parameter and return a list of label values and a list of scores.
	"""
	def classify_multiple_images(self,imgs):
		
		neighbors = np.array([])
		label_index = np.array([])
		label_value = np.array([])

		for x in imgs:
			l,n = self.classify_single_image(x)
			neighbors = np.append(neighbors,n)
			label_index = np.append(label_index,l)

		label_index = label_index.astype(int)

		for x in label_index:
			label_value = np.append(label_value, self.train_label[x])


		label_index = label_index.tolist()
		neighbors = neighbors.tolist()

		return label_index , neighbors










