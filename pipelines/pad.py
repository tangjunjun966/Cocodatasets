

import numpy as np

class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def impad(self,img, shape, pad_val=0):
        """
        在左上角pad填充图像，因此不需要调整box
        Pad an img to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple): Expected padding shape.
            pad_val (number or sequence): Values to be filled in padding areas.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + (img.shape[-1],)
        assert len(shape) == len(img.shape)
        for i in range(len(shape) - 1):
            assert shape[i] >= img.shape[i]
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[:img.shape[0], :img.shape[1], ...] = img
        return pad

    def impad_to_multiple(self,img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.

        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val (number or sequence): Same as :func:`impad`.

        Returns:
            ndarray: The padded image.
        """
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.impad(img, (pad_h, pad_w), pad_val)

    def _pad_img(self, results):

        if self.size is not None:
            padded_img = self.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = self.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape  # 更改pad尺寸
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        if results['with_mask']:
            padded_masks = [
                self.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results['gt_masks']
            ]
            results['gt_masks'] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        if results['mode'] in ['train','val']:
            self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str

