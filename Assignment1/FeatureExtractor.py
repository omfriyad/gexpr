
class FeatureExtractor:

	def __init__(self, img):
		self.img = img
	
	def extract_feature(self):
		return self.img


	""" 
	implement the following sub-classes of FeatureExtractor
		1. ColourHistogramExtractor
		2. HoGExtractor
		3. SIFTBoVWExtractor
	"""