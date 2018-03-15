import numpy as np
from mnist import train_images, train_labels, test_images, test_labels

"""
   ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``
"""


def training_data():
    load_data = [train_images(), train_labels()]
    training_inputs = [np.reshape(x, (784, 1)) for x in load_data[0]]
    training_results = [vectorized_result(y) for y in load_data[1]]
    train = zip(training_inputs, training_results)
    return train


"""
    validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

"""


def validation_data():
    load_data = test_images(), test_labels()
    validation_inputs = [np.reshape(x, (784, 1)) for x in load_data[0]]
    validate = zip(validation_inputs, load_data[1])
    return validate


def test_data():
    load_data = [test_images(), test_labels()]
    test_inputs = [np.reshape(x, (784, 1)) for x in load_data[0]]
    test = zip(test_inputs, load_data[1])
    return test


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
