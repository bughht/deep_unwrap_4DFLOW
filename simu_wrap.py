import numpy as np


def wrap(img, venc_reduction=0.5):
    """
    Simulate wrap venc reduced phase image to [-pi, pi] range.
    Args:
        img: phase image
        venc_reduction: VENC reduction factor
    Returns:
        wrapped phase image
    """
    img=img.copy()
    img[1:] /= venc_reduction
    img[1:] = (img[1:]+np.pi) % (2*np.pi)-np.pi
    return img
