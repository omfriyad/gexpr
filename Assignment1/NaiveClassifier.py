from FeatureExtractor import FeatureExtractor

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
	def __init__(self, trainImg, trainLabel, feature):
	    self.trainImg = trainImg
	    self.trainLabel = trainLabel
	    self.feature = feature


	""" 
	extract feature from single image, which should accept an image 
	as pa- rameter. The image is a (number of channels, x dim, y dim) numpy array. 
	This method should return the feature extracted by calling the 
	extract feature method of the FeatureExtractor object.
	"""
	def extract_feature_from_single_image(self):
		pass





	"""
	extract feature from multiple images, which should accept a 
	(number of images, number of channels, x dim, y dim) numpy array 
	as parameter and return a (number of images, feature dim) array.
	"""
	def extract_feature_from_single_image(self):
		pass





	"""
	classify single image, which will accept an image as parameter 
	and return a label value and the corresponding score.
	"""
	def classify_single_image(self):
		pass 





	"""
	classify multiple images, which will accept multiple images as 
	parameter and return a list of label values and a list of scores.
	"""
	def classify_multiple_image(self):
		pass 







