
class ClassificationMetrices:

	"""
	1. __init__ :  
		This method should accept a list of training labels, 
		a list of predicted labels, and a list of predicted scores 
		as parameter. It should also construct the confusion matrix 
		and cache this information.
	"""
	def __init__(self, train_label, test_label,predicted_score):
		self.train_label = train_label
		self.test_label = test_label
		self.predicted_score = predicted_score


	"""
	2. get_confusion_matrix_for_heatmap: 
		Returns a 0−1 normalized confusion matrix.
	"""
	def get_confusion_matrix_for_heatmap(self):
		pass



	"""
	3. calculate_accuracy: 
		Calculate and return the accuracy of the predictions. 
		It should return a value in the range 0 − 1.
	"""
	def calculate_accuracy(self):
		pass




	"""
	4. calculate_precision:
		Calculate and return the average precision. 
	"""
	def calculate_precision(self):
		pass



	"""
	5. calculate_recall:
		Calculate and return the average recall. 
	"""
	def calculate_recall(self):
		pass



	"""
	6. calculate_f1:
		Calculate and return the F1 score.
	"""
	def calculate_f1(self):
		pass




	"""
	7. calculate_roc_values:
		This method should accept a list of thresholds as parameter and 
		then calculate and return the ROC values in a list of lists. 
		A list per threshold value per class.
	"""
	def calculate_roc_values(self):
		pass



	"""
	8. calculate_lift_values: 
		Similar to calculate roc values, but calculates lift values.
	"""
	def calculate_lift_values(self):
		pass






