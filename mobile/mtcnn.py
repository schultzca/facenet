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
    return cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)


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
    max_exp = int(math.log(12 / (a * length), b))

    # Compute scales using maximum exponent value.
    return [a * b ** exp for exp in range(0, max_exp + 1)]


def generate_bbox(face_cls, bbox_reg, scale, threshold):
    """
    Compute bounding box coordinates using face classification and bounding
    box regression output from proposal network. The output of the network
    is down sampled from the original image so the results are rescaled with
    respect to the original image dimensions.

    :param face_cls: face detection output from proposal network
    :param bbox_reg: bounding box regression output from proposal network
    :param scale: image scale
    :param threshold: face detection score threshold
    :return: bounding box array [x1, y1, x2, y2, score, dx1, dy1, dx2, dy2]
    """

    # Find x and y coordinates that contain a face given the threshold.
    x, y = np.where(face_cls[0, :, :, 1] > threshold)

    # Return empty array if there are no positive classifications that
    # meat the threshold requirement.
    if x.size == 0:
        return np.empty((0, 9))

    # Filter the regression results to only include the classified face
    # coordinates.
    bbox_reg_filtered = bbox_reg[0, x, y, :]

    # Scale bounding box coordinates back to original image dimensions.
    bb = np.vstack((x, y)).T
    q1 = np.fix((2 * bb + 1) / scale)
    q2 = np.fix((2 * bb + 12 - 1 + 1) / scale)

    # (x1, y1, x2, y2, score, dx1, dy1, dx2, dy2)
    bbox = np.hstack([q1, q2, np.expand_dims(face_cls[0, x, y, 1], 1), bbox_reg_filtered])

    return bbox


def non_max_suppression(boxes, overlap_threshold, index=None):
    """
    Apply non-maximum suppression to bounding box candidates. This method
    iterates over each bounding box and computes the intersection over
    union with respect to each remaining candidate. Candidates that have a
    large intersection over union are pruned. The process repeats until
    there are no remaining candidates.

    The order in which you iterate over the bounding box candidates is
    important since overlapping candidates are removed. This is why the
    index argument is provided. This allows you to prioritize the order
    in which bounding boxes are processed. For example if in addition to
    bounding box coordinates you have an associated classification score
    you will likely want to prioritize bounding boxes with the highest
    classification score.

    :param boxes: bounding box array [?, x1, y1, x2, y2]
    :param overlap_threshold: intersection over union threshold
    :param index: (optional) index priority
    :return: filtered bounding box array
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.empty((0, 9))

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if index is not None:
        idxs = np.argsort(index)
    else:
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        intersection = w * h
        overlap = intersection / (area[i] + area[idxs[:last]] - intersection)

        # delete all indexes from the index list that are over overlap threshold
        idxs = idxs[np.where(overlap <= overlap_threshold)]

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def first_stage(image, pnet):

    scales = scale_pyramid(image, b=0.709, min_length=20)

    # Store bounding boxes generated at each scale.
    all_bboxes = np.empty((0,9))

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
        bbox_reg, face_cls = pnet(image_data)

        # Convert back to normal coordinates from Caffe coordinates.
        bbox_reg = np.transpose(bbox_reg, (0, 2, 1, 3))
        face_cls = np.transpose(face_cls, (0, 2, 1, 3))

        # Compute bounding box coordinates from classification and regression
        # outputs from proposal network.
        bbox_proposals = generate_bbox(face_cls, bbox_reg, scale, 0.6)

        # Apply non-maximum suppression to consolidate bounding box proposals.
        bboxes = non_max_suppression(bbox_proposals, 0.5, bbox_proposals[:, 4])

        # Store bounding boxes from this scale.
        all_bboxes = np.append(all_bboxes, bboxes, axis=0)

    return all_bboxes

if __name__ == "__main__":

    # Read example image
    image = cv2.imread("../data/images/Anthony_Hopkins_0001.jpg")

    # Convert color space to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN graph
    with tf.Session() as sess:

        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        box_org = detect_face.detect_face(image.copy(), 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)

        box_new = first_stage(image.copy(), pnet)





























