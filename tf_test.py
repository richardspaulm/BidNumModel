from __future__ import print_function

import math

# from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.python.data import Dataset
import re

import seaborn as sns
import glob
from sklearn.preprocessing import MinMaxScaler
from format_for_training import format_df


data = pd.read_csv("test_format.csv")
data = format_df(data)

scaler = MinMaxScaler()

def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)


tf.logging.set_verbosity(tf.logging.FATAL)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


dataset = data
# dataset = dataset.drop([2060062])
dataset.label = dataset.label.astype(int)
print(dataset.head())
print(dataset.describe())

dataset = dataset.reindex(np.random.permutation(dataset.index))
# exit()
# print(data)


def preprocess_features(df):
  
  preprocess = df.copy()
  preprocess = preprocess.drop(['label'], axis=1)
  # Create a synthetic feature.
  return preprocess

def preprocess_targets(df):
  output_targets = pd.DataFrame()
  output_targets["label"] = (df["label"])
  return output_targets


def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
    processed_features = examples_dataframe
    processed_features["Length_Data"] = linear_scale(examples_dataframe["Length_Data"])
    processed_features["Number_Numeric"] = linear_scale(examples_dataframe["Number_Numeric"])
    processed_features["Number_Alpha"] = linear_scale(examples_dataframe["Number_Alpha"])
    processed_features["Special_Char"] = linear_scale(examples_dataframe["Special_Char"])
    processed_features["Space_Count"] = linear_scale(examples_dataframe["Space_Count"])
    processed_features["RFP_count"] = linear_scale(examples_dataframe["RFP_count"])
    processed_features["RFT_count"] = linear_scale(examples_dataframe["RFT_count"])
    processed_features["IFB_count"] = linear_scale(examples_dataframe["IFB_count"])
    processed_features["ITB_count"] = linear_scale(examples_dataframe["ITB_count"])
    processed_features["ITP_count"] = linear_scale(examples_dataframe["ITP_count"])
    processed_features["Slash_count"] = linear_scale(examples_dataframe["Slash_count"])
    return processed_features


# training_examples = preprocess_features(dataset.head(300))
print(dataset.head())
training_targets = preprocess_targets(dataset.head(100000)) 
validation_targets = preprocess_targets(dataset.tail(20000))

print("Normalizing Data")
# training_examples = normalize_linear_scale(preprocess_features(dataset.head(100000)))
# validation_examples = normalize_linear_scale(preprocess_features(dataset.tail(20000)))
training_examples = preprocess_features(dataset.head(100000))
validation_examples = preprocess_features(dataset.tail(20000))
dataset = None


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(1)


    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps / periods
    
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = lambda: my_input_fn(
        training_examples, training_targets, batch_size)


    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 1*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 1      # Retain the 10 most recent checkpoints.
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(training_examples),
        n_classes=2,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        model_dir="./model_checkpoints",
        config=my_checkpointing_config
    )
    print("Training Model")
    print("LogLoss error")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(input_fn=training_input_fn,
        steps=steps_per_period)
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,2)
            
        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,2)  
        
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
        _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    print("Model training finished.")
    # Remove event files to save disk space.

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])


    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.savefig("LogLoss.png")

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion.png")

    return classifier

classifier = train_nn_classification_model(
    learning_rate=0.013,
    steps=50000,
    batch_size=256,
    hidden_units=[10, 10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


























