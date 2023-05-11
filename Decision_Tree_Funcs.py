import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """
    import scipy.stats as stats

    feature_vals = np.unique(x_inputs[:, feature_index])

    indexDict = {}
    for key in feature_vals:
        indexDict[key] = [i for i, x_i in enumerate(x_inputs) if x_i[feature_index] == key]

    classes_by_feature = {key: stats.mode(y_outputs[indexDict[key]], keepdims=True).mode for key in feature_vals} 

    total_correct = 0
    for i in range(x_inputs.shape[0]):
        if classes_by_feature[x_inputs[i, feature_index]] == y_outputs[i]:
            total_correct += 1

    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """
    import sys
    classes = np.array(classes)
    counts = np.sum(y_outputs[:, None] == classes[None, :], axis = 0)
    percents = 1.0 * (counts / np.sum(counts))
    entropy = -np.dot(percents, np.log2(percents + sys.float_info.min))

    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """Calculate the information remainder after splitting the input-output set based on the
given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    # set_entropies = np.zeros(len(classes))
    # for index, feature in enumerate(classes):
    #     indices = [i for i, y_i in enumerate(y_outputs) if y_i == classes[index]]
    #     set_entropies[index] = set_entropy(x_inputs[indices], y_outputs[indices], classes)

    # set_entropies = np.zeros(2)
    # true_indices = [i for i, y_i in enumerate(y_outputs) if y_i == feature_index]
    # false_indices = [[i for i, y_i in enumerate(y_outputs) if y_i != feature_index]]
    # set_entropies[true_indices] = set_entropy(x_inputs[true_indices], y_outputs[true_indices], classes)
    # set_entropies[false_indices] = set_entropy(x_inputs[false_indices], y_outputs[false_indices], classes)
    
    
    class_vals = np.unique(x_inputs[:, feature_index])

    indexDict = {}
    for key in class_vals:
        indexDict[key] = [i for i, x_i in enumerate(x_inputs) if x_inputs[i, feature_index] == key]

    set_entropies = [set_entropy(x_inputs[indexDict[key]], y_outputs[indexDict[key]], classes) for key in class_vals]

    # Calculate the remainder
    n_samples = len(x_inputs)
    sums = np.array([len(x_inputs[indexDict[key]]) for key in class_vals]) / n_samples
    remainder = np.sum(np.dot(sums, set_entropies))

    gain = overall_entropy - remainder

    return gain
