import numpy as np

import cv2


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def imnormalize(self, img, mean, std, to_rgb=True):
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32)
        return (img - mean) / std

    def __call__(self, results):
        results['img'] = self.imnormalize(results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str
