import numpy as np
import cv2


class Resize(object):
    """
    Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self, img_scale=None, img_interpolation='bilinear'):

        assert img_scale is not None, 'img_scale must be value,eg: list int tuple'
        self.img_scale = img_scale
        self.img_interpolation = img_interpolation

    @staticmethod
    def random_sample(img_scales):
        scale_h = [s[0] for s in img_scales]
        scale_w = [s[1] for s in img_scales]
        img_scale_h = np.random.randint(min(scale_h), max(scale_h) + 1)
        img_scale_w = np.random.randint(min(scale_w), max(scale_w) + 1)
        scale = (img_scale_h, img_scale_w)
        return scale

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if isinstance(self.img_scale, (list, tuple)):
            if isinstance(self.img_scale[0], (list, tuple)):
                scale = self.random_sample(self.img_scale)
            elif isinstance(self.img_scale[0], int):
                scale = (self.img_scale[0], self.img_scale[1])
            else:
                raise TypeError('type of self.img_scale must be  list  int tuple,but self.img_scale:{}'.format(
                    type(self.img_scale)))
        elif isinstance(self.img_scale, int):
            scale = (self.img_scale, self.img_scale)
        else:
            raise TypeError('img_scale must be value,eg: list  int tuple')
        assert len(scale) == 2, 'length of  scale geted must equal 2 ,but scale:{} '.format(scale)
        results['scale'] = scale  # resize 后的[h,w]即[y,x]
        return scale

    def _resize_img(self, results, interpolation='bilinear'):
        resize_img = imrescale(results['img'], results['scale'], interpolation=interpolation)
        results['img'] = resize_img
        results['img_shape'] = resize_img.shape  # resize后的图片大小
        results['pad_shape'] = resize_img.shape  # in case that there is no padding

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        ori_shape = results['ori_shape']
        bboxes = results['ori_bboxes'].copy()
        h_factor, w_factor = img_shape[0] / ori_shape[0], img_shape[1] / ori_shape[1]
        bboxes[:, 0::2] = bboxes[:, 0::2] * w_factor
        bboxes[:, 1::2] = bboxes[:, 1::2] * h_factor
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
        results['gt_bboxes'] = bboxes

    def _resize_masks(self, results):
        if results['with_mask']:
            masks = [
                imrescale(mask, results['scale'], interpolation='nearest')
                for mask in results['ori_masks']]
            results['gt_masks'] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)  # 得到新的scale
        self._resize_img(results, interpolation=self.img_interpolation)
        if results['mode'] in ['train','val']:
            self._resize_bboxes(results)
            self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__

        return repr_str


def imrescale(img, scale, interpolation='bilinear'):
    '''
    :param img: 原始图片
    :param scale: 新图片resize尺寸，[h,w]即[y,x]
    :param interpolation: resize 方法
    :return:
    '''
    h_size, w_size = scale[0], scale[1]
    scale_size = (w_size, h_size)  # resize 需转换成[w,h]才能匹对
    resized_img = cv2.resize(img, scale_size, interpolation=interp_codes[interpolation])
    return resized_img


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
