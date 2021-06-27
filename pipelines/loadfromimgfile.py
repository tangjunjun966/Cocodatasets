import os
import cv2
import numpy as np
import six


class LoadImageFromFile(object):

    def __init__(self, to_float32=False, flags=-1):
        self.to_float32 = to_float32
        self.flags = flags

    def __call__(self, results):
        img_root = results['img_info']['img_root']
        if isinstance(img_root, np.ndarray):
            img = img_root
        elif isinstance(img_root, six.string_types):
            if not os.path.isfile(img_root):
                raise FileNotFoundError('loading img file does not exist: {}'.format(img_root))


            img = cv2.imread(img_root, flags=self.flags)
        else:
            raise TypeError('"img" must be a numpy array or a img path ')

        if self.to_float32:
            img = img.astype(np.float32)
        results['img_root'] = img_root
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(self.to_float32)
