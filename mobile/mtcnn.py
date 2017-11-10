import cv2
import math
import numpy as np
import src.align.detect_face as detect_face
import tensorflow as tf


def resize(image, height, width):
    """
    Resize image using bilinear interpolation. This is the default for the
    opencv method resize.

    :param image: input image
    :param height: new height
    :param width: new width
    :return: scaled image
    """
    return cv2.resize(image, (height, width))


def normalize_image(image):
    """
    Normalize pixel values to values in range (-1 to 1). The input image
    should have 8-bit pixels with values between 0 and 255. To normalize
    we first change the center to zero and then we scale.

    :param image: input image
    :return: normalized image
    """
    return (image - 127.5) * 0.0078125


def scale_pyramid(image, a=None, b=0.5, min_length=12):
    """
    This function computes a list of exponentially decreasing scale factors
    until the original image reaches a minimum dimension limit.

        scale = a * b ^ exponent for (exponent in [0, -1, -2, -3, ...]

    The minimum scaled image must have a length greater than or equal to the
    minimum length.

    :param image: image input
    :param a: proportional factor (defaults to 12 / min_length)
    :param b: base of exponential
    :param min_length: minimum scaled distance limit
    :return: (list) scale factors
    """
    # Get image dimensions
    height, width, _ = image.shape

    # Select smallest dimension of image
    length = np.amin([height, width])

    # Compute "a" in equation above. (Not sure why 12 is hardcoded.)
    if a is None:
        a = 12. / min_length

    # Compute maximum exponent that meets minimum length requirement.
    # Given this inequality, length * a * b ^ x >= minimum length, we can
    # compute maximum exponent x >= log_b (minimum length / (a * length))
    # given condition.
    max_exp = int(math.log(min_length / (a * length), b))

    # Compute scales using maximum exponent value.
    return [a * b ** exp for exp in range(1, max_exp + 1)]


def first_stage(image, pnet):

    scales = scale_pyramid(image, b=0.709, min_length=20)

    for scale in scales:

        # Get scaled dimensions
        hs, ws, _ = np.ceil(scale * np.array(image.shape)).astype(np.int)

        # Scale and normalize image
        image_data = normalize_image(resize(image, hs, ws))

        # PNet is designed to process a batch of images at once so we
        # need to add that dimension.
        image_data = np.expand_dims(image_data, 0)

        # PNet was trained using Caffe so we need to swap the x and y
        # coordinates of the image.
        image_data = np.transpose(image_data, (0, 2, 1, 3))

        # Run image through proposal network which returns the bounding
        # box regression and face classification results.
        reg, cla = pnet(image_data)

        # Convert back to normal coordinates from Caffe coordinates.
        reg = np.transpose(reg, (0, 2, 1, 3))
        cla = np.transpose(cla, (0, 2, 1, 3))


if __name__ == "__main__":

    # Read example image
    image = cv2.imread("../data/images/Anthony_Hopkins_0001.jpg")

    # Convert color space to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN graph
    with tf.Session() as sess:

        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        first_stage(image, pnet)



























