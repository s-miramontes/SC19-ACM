import time
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from skimage import color, io, morphology, img_as_ubyte, feature, exposure, filters, util
from skimage.filters import frangi, hessian
import pandas as pd
from skimage import exposure
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist

# some more dependencies
import tensorflow as tf

from keras import models, layers
import keras
from skimage import transform
from skimage.filters import gaussian

from numpy import loadtxt
from keras.models import load_model
from skimage.morphology import label, binary_erosion
from skimage.measure import regionprops



# STARTING MIRRORED STRATEGY: aka parallelizing in GPUs
mirrored_strat = tf.distribute.MirroredStrategy(devices=['/gpu:6', '/gpu:7'])



# Loading Image Crops
# That is , we are assuming that the images are already generated

# 1. Obtain a list of files (each file is an image cross-seciont within the stack)
# 2. Sort this list to preserve order of cross-sections in the stack

# ---------------------------------------------------------------------------------

# Aux Functions
# Function to sort files
# label creation for binary data
def create_labels(ones_size, zeros_size):
    """
    Generation of class labels accroding
    to given sizes:

    input1: length/number of 1's desired
    input2: length/number of 0's desired
    returns: complete list of labels accor-
    ding to the specifications on input
    """

    the_ones = np.ones(ones_size).tolist()
    the_zeros = np.zeros(zeros_size).tolist()
    concat_labels = the_ones + the_zeros

    return concat_labels

def make_dataframe(list_of_imgs, ones, zeros):
    """
    Creates data frame from a list of images
    and adds appropriate labels based on
    sizes (ones and zeros)

    input 1: list of images
    input 2: number of 1's
    input 3: number of 0's

    returns: dataframe with labels
    """
    frame = pd.DataFrame(list_of_imgs)
    # attach labels according to sizes (ones and zeros)
    frame['Labels'] = create_labels(ones, zeros)

    return frame


def sort_files(list_files):
    """
    Sorts file list (image slices)
    to preserve order.

    list_files: list of files from
    os.listdir
    returns: list of files.
    """
    sort_list = sorted(map(int,list_files))
    to_str = list(map(str,sort_list))

    return to_str

def unflatten_values(unlabeled_dataframe):
    """
    Takes the unlabeled dataframe's values,
    to unflatten them to (16,16) imgs
    inside a list. Returns list.

    input 1: dataframe with no labels
    output: list of unflattened images
    """
    values = unlabeled_dataframe.values
    temp_unflat = []
    for element in values:
        un = element.reshape(16,16)
        temp_unflat.append(un)

    return temp_unflat


def matrix_shuffle(list_1, list_2, ones, zeros, separate_files=False):
    """
    Uses lists on input to create dataframes
    with appropriate labels. Concatenates
    these dataframes as a way to build our X
    and y prior to training and testing.
    Shuffles the matrix (X) to introduce random-
    ness in test/training process.

    input 1: list of imgs
    input 2: list of imgs (another one)
    input 3: number of 1's
    input 4: number of 0's

    output: shuffled matrix (no labels), labels
    """

    # creates dataframe with labels
    if separate_files:
        frame_1 = make_dataframe(list_1, ones, 0)
        frame_2 = make_dataframe(list_2, 0, zeros)
    else:
        frame_1 = make_dataframe(list_1, ones, zeros)
        frame_2 = make_dataframe(list_2, ones, zeros)

    list_frames = [frame_1, frame_2]

    # double check here that the labels are kept as they should,
    # it works!
    attach_them = pd.concat(list_frames, ignore_index=True)

    # shuffle it!
    attached_shuffle = attach_them.sample(frac=1).reset_index(drop=True)

    # drop the labels but save-them!
    y_labels = attached_shuffle['Labels']
    attached_no_labels = attached_shuffle.drop(['Labels'], axis=1)

    # returns shuffled matrix without labels, and its dropped labels
    return attached_no_labels, y_labels


def conf_matrix(y, predictions):
    cm = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1), labels=[0, 1])
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    TN = cm[0, 0]

    # Sensitivity, hit rate, recall, or true positive rat
    TPR = TP/(TP+FN)
        # Specificity or true negative rate
    TNR = TN/(TN+FP)
        # Precision or positive predictive value
    PPV = TP/(TP+FP)
        # Negative predictive value
    NPV = TN/(TN+FN)
        # Fall out or false positive rate
    FPR = FP/(FP+TN)
        # False negative rate
    FNR = FN/(TP+FN)
        # False discovery rate
    FDR = FP/(TP+FP)

        # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    all_results = {'Sensitivity': TPR, 'Specificity': TNR, 'Precision': PPV,
                  'Neg. Pred Rate': NPV, 'False Pos. Rate': FPR,
                  'False Neg. Rate': FNR, 'False Discovery Rate:': FDR,
                  'Accuracy': ACC}

    frame_results = pd.DataFrame(list(all_results.items()), columns=['Measure', 'Rates'])

    #return frame_results, cm, (TP, FN, FP, TN)
    return all_results


def metrics_dataframe(your_list):
    """
    Converts list of dictionaries to
    data frame ensuring order in list.
    (since lists preserve order)

    your_list : list of dictionaries
    output: dataframe of however many
    rows (elements) in your_list.
    """

    size = len(your_list)
    all_mets_df = pd.DataFrame()

    for dic in range(size):
        temp_df = pd.DataFrame(your_list[dic], index=[0])
        all_mets_df = all_mets_df.append(temp_df,
                                         ignore_index=True)

    return all_mets_df


def predict_imgs(fibs, voids, cnn_model, rotate=False):

    if rotate:
        fib_resize = [transform.rotate(transform.resize(rgb2gray(a),
                                                        (16, 16),
                                                        anti_aliasing=True),
                                       angle=90).flatten() for a in fibs]
        void_resize = [transform.rotate(transform.resize(rgb2gray(b),
                                                         (16, 16),
                                                         anti_aliasing=True),
                                        angle=90).flatten() for b in voids]
    else:
        fib_resize = [transform.resize(rgb2gray(a),(16,16), anti_aliasing=True).flatten() for a in fibs]
        void_resize = [transform.resize(rgb2gray(b), (16,16), anti_aliasing=True).flatten() for b in voids]

    shuf_imgs, shuf_labels = matrix_shuffle(fib_resize,
                                            void_resize,
                                            len(fib_resize),
                                            len(void_resize),
                                            separate_files=True)

    X = np.asarray(unflatten_values(shuf_imgs)).astype('float32')
    X = X.reshape(X.shape[0], 16, 16, 1)

    y = to_categorical(shuf_labels)

    # making the prediction
    pred = cnn_model.predict(X)
    pred = (pred > 0.5)

    # get measurements: grades is a dictionary
    results = conf_matrix(y, pred)

    return results
# --------------------------------------------------------


# Paths to the cropped images

fiber_path = 'ready_data_first_stack/gtn/crops_per_slice/'
void_path = 'ready_data_first_stack/gtn/voids_per_slice/'

# Listed Directory names containing image crops of voids and fibs
fiber_crops = os.listdir(fiber_path)
fiber_crops = sort_files(fiber_crops)
void_spaces = os.listdir(void_path)
void_spaces = sort_files(void_spaces)

# Load the CNN model and display information
model = load_model("LeNet/MODEL/model-on-nats.h5")
print("Below is the Summary of the already trained CNN model: ")
model.summary()

print("\n Here is the performance of the model per epoch")
# performance_isvc = plt.imread("LeNetPerformance-ISVC.png")
# fig, ax = plt.subplots(figsize=(20,15))
# ax.imshow(performance_isvc)

# ---------------------------------------------------------------------------------


# Begin with classification

# generating the 10% indexing
with mirrored_strat.scope():
    every_100th = np.arange(0, 1000, 10)

    # we start here
    metrics_reg = []

    time_start_reg = time.time()

    print("Classifying...")
    for i in every_100th:
        fib_collection = io.ImageCollection(os.path.join(fiber_path,
                                                         fiber_crops[i],
                                                         '*.png'),
                                            conserve_memory=True)
        void_collection = io.ImageCollection(os.path.join(void_path,
                                                          void_spaces[i],
                                                          '*.png'),
                                             conserve_memory=True)

        classify = predict_imgs(fib_collection, void_collection, model)
        metrics_reg.append(classify)

    time_end_reg = time.time()

    msg = "Done! The process took " + str(time_end_reg - time_start_reg)
    print(msg)

    # Creating dataframe with the desired results and save
    df_reg = metrics_dataframe(metrics_reg)
    df_reg.to_csv("classification_metrics.csv")

    print("Below are mean metrics for the classification results\n")
    print(df_reg.mean(axis=0))
    print('\n')
    print(df_reg)
