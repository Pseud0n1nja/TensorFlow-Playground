import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np


# Load datasets.
#Data Scource 
#http://download.tensorflow.org/data/iris_training.csv
#http://download.tensorflow.org/data/iris_test.csv﻿

training_set = base.load_csv_with_header(filename = "iris_training.csv",
                                         features_dtype=np.float32,
                                         target_dtype=np.int)

test_set = base.load_csv_with_header(filename= "iris_test.csv",
                                     features_dtype=np.float32,
                                     target_dtype=np.int)

# Model creation

# Specifying features (real-value data)
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

#Using/Inheriting Linear Classifier from Estimator

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="/Users/iris_model")

#Defining Input function
def input_fn(dataset):
    def func():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return func


# raw data -> input function -> feature columns -> model

### Training

classifier.train(input_fn=input_fn(training_set),
               steps=1000)

### Evaluation

accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), 
                                     steps=100)["accuracy"]

# Evaluate accuracy.
print('\nAccuracy: {0:f}'.format(accuracy_score))


