import cv2
import numpy as np


def guided_filter(image, mask, radius=5, eps=10):
    """
    :param image: np.ndarray base image (3 or 1 channel)
    :param mask: 3-channel np.ndarray mask
    :param radius: radius of Guided Filter
    :param eps: regularization term of Guided Filter.
    :return: soft segmentation mask
    """
    if np.shape(image)[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.ximgproc.guidedFilter(guide=image, src=mask, radius=radius, eps=eps, dDepth=-1)
