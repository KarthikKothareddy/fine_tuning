# import the necessary packages
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense


class FCHeadNet:
	@staticmethod
	def build(baseModel, classes, D):
		"""
		Initialize the head model that will be placed on top of
		the base, then add a FC layer
		:param baseModel: base trained model
		:param classes: number of classes
		:param D: Number of layers in Dense layer
		:return:
		"""
		headModel = baseModel.output
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(D, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)

		# add a softmax layer
		headModel = Dense(classes, activation="softmax")(headModel)

		# return the model
		return headModel
