import tensorflow as tf
from tensorflow import keras

def preprocess(image, label):
	resized_image = tf.image.resize(image, [224, 224])
	final_image = keras.applications.efficientnet.preprocess_input(resized_image)
	return final_image, label