
import numpy as np


class ClassificationMetrices:

	"""
	1. __init__ :  
		This method should accept a list of training labels, 
		a list of predicted labels, and a list of predicted scores 
		as parameter. It should also construct the confusion matrix 
		and cache this information.
	"""
	def __init__(self, train_label, predicted_label,predicted_score):
		
		self.train_label = train_label
		self.predicted_label = predicted_label
		self.predicted_score = predicted_score

		temp_matrix = np.unique(self.train_label)
		size = len(np.unique(self.train_label))
		self.matrix = np.zeros([ size, size ])

		self.train_label = self.train_label - temp_matrix[0]
		self.predicted_label = self.predicted_label - temp_matrix[0]

		for x in range(len(self.train_label)):
			i = self.train_label[x]
			j = self.predicted_label[x]
			self.matrix[i][j] += 1

	"""
	2. get_confusion_matrix_for_heatmap: 
		Returns a 0−1 normalized confusion matrix.
	"""
	def get_confusion_matrix_for_heatmap(self):

		normalized_confusion_matrix = self.matrix / self.matrix.sum(axis=1)
		self.matrix = normalized_confusion_matrix
		return normalized_confusion_matrix


	"""
	3. calculate_accuracy: 
		Calculate and return the accuracy of the predictions. 
		It should return a value in the range 0 − 1.
	"""
	def calculate_accuracy(self):
		
		total_sum = np.sum(self.matrix)							#sum of whole matrix
		diag_sum = np.sum(np.diag(self.matrix))					#sum of diagonal matrix
		accuracy = diag_sum/ total_sum							#getting accuracy

		return accuracy

	"""
	4. calculate_precision:
		Calculate and return the average precision. 
	"""
	def calculate_precision(self):

		precision = np.diag(self.matrix) / np.sum(self.matrix, axis = 0)
		average_precision = np.mean(precision)

		return average_precision


	"""
	5. calculate_recall:
		Calculate and return the average recall. 
	"""
	def calculate_recall(self):

		recall = np.diag(self.matrix) / np.sum(self.matrix, axis = 1)
		average_recall = np.mean(recall)

		return average_recall

	"""
	6. calculate_f1:
		Calculate and return the F1 score.
	"""
	def calculate_f1(self):
		
		# f1 = (2PR)/ (P+R)
		precision = self.calculate_precision()
		recall = self.calculate_recall()

		f1 = (2*precision*recall) / (precision+recall)

		return f1


	"""
	7. calculate_roc_values:
		This method should accept a list of thresholds as parameter and 
		then calculate and return the ROC values in a list of lists. 
		A list per threshold value per class.
	"""
	def calculate_roc_values(self,thresholds):
		pass



	"""
	8. calculate_lift_values: 
		Similar to calculate roc values, but calculates lift values.
	"""
	def calculate_lift_values(self):
		pass






