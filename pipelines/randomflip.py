import numpy as np


class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical'], \
            'flip direction must be horizontal or vertical,but direction:{}'.format(direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def imflip(self, img, direction='horizontal'):
        """Flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or "vertical".

        Returns:
            ndarray: The flipped image.
        """
        assert direction in ['horizontal', 'vertical']
        if direction == 'horizontal':
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=0)

    def __call__(self, results):

        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            if 'flip_direction' not in results:
                results['flip_direction'] = self.direction

                # flip image
            results['img'] = self.imflip(results['img'], direction=results['flip_direction'])
            # flip bboxes
            if results['mode'] in ['train', 'val']:
                results['gt_bboxes'] = self.bbox_flip(results['gt_bboxes'],
                                                      results['img_shape'],
                                                      results['flip_direction'])
                # flip masks
                if results['with_mask']:
                    results['gt_masks'] = [
                        self.imflip(mask, direction=results['flip_direction'])
                        for mask in results['gt_masks']
                    ]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(self.flip_ratio)
