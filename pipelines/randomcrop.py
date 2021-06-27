import numpy as np


class RandomCrop(object):
    """
    Random crop the image & bboxes & masks.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size=None, margin_size=None, crop_ratio=0.5):
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.margin_size = margin_size

    def crop_size_func(self, results):
        flip = True if np.random.rand() < self.crop_ratio else False
        if flip:
            img = results['img']
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            crop_x1, crop_y1 = np.max(crop_x1, 0), np.max(crop_y1, 0)
            crop_x2, crop_y2 = np.min(crop_x2, img.shape[1]), np.min(crop_y2, img.shape[0])

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape

            if results['mode'] in ['train', 'val']:
                # crop bboxes accordingly and clip to the image boundary

                bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
                bboxes = results['gt_bboxes'] - bbox_offset  # 只需减去左边即可
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
                results['gt_bboxes'] = bboxes
                # filter out the gt bboxes that are completely cropped
                if 'gt_bboxes' in results:
                    gt_bboxes = results['gt_bboxes']
                    valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (gt_bboxes[:, 3] > gt_bboxes[:, 1])
                    # if no gt bbox remains after cropping, just skip this image
                    if not np.any(valid_inds):
                        return None
                    results['gt_bboxes'] = gt_bboxes[valid_inds, :]
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][valid_inds]

                    # filter and crop the masks
                    if 'gt_masks' in results:
                        valid_gt_masks = []
                        for i in np.where(valid_inds)[0]:
                            gt_mask = results['gt_masks'][i][crop_y1:crop_y2, crop_x1:crop_x2]
                            valid_gt_masks.append(gt_mask)
                        results['gt_masks'] = valid_gt_masks
        return results

    def margin_size_func(self, results):
        # 边缘随机选择
        flip = True if np.random.rand() < self.crop_ratio else False
        if flip:
            img = results['img']
            margin_h, margin_w = self.margin_size[0], self.margin_size[1]
            assert margin_h <= img.shape[0] and margin_w <= img.shape[1], 'margin_size must be less than img_shape'
            crop_x1 = np.random.randint(0, margin_w + 1)
            crop_y1 = np.random.randint(0, margin_h + 1)
            crop_x2 = img.shape[1] - np.random.randint(0, margin_w + 1)
            crop_y2 = img.shape[0] - np.random.randint(0, margin_h + 1)

            offset_w, offset_h = crop_x1, crop_y1

            crop_x1, crop_y1 = np.max([crop_x1, 0]), np.max([crop_y1, 0])
            crop_x2, crop_y2 = np.min([crop_x2, img.shape[1]]), np.min([crop_y2, img.shape[0]])

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape

            if results['mode'] in ['train', 'val']:
                # crop bboxes accordingly and clip to the image boundary

                bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
                bboxes = results['gt_bboxes'] - bbox_offset  # 只需减去左边即可
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
                results['gt_bboxes'] = bboxes
                # filter out the gt bboxes that are completely cropped
                if 'gt_bboxes' in results:
                    gt_bboxes = results['gt_bboxes']
                    valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (gt_bboxes[:, 3] > gt_bboxes[:, 1])
                    # if no gt bbox remains after cropping, just skip this image
                    if not np.any(valid_inds):
                        return None
                    results['gt_bboxes'] = gt_bboxes[valid_inds, :]
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][valid_inds]

                    # filter and crop the masks
                    if 'gt_masks' in results:
                        valid_gt_masks = []
                        for i in np.where(valid_inds)[0]:
                            gt_mask = results['gt_masks'][i][crop_y1:crop_y2, crop_x1:crop_x2]
                            valid_gt_masks.append(gt_mask)
                        results['gt_masks'] = valid_gt_masks
        return results

    def __call__(self, results):

        if self.crop_size is not None:
            results = self.crop_size_func(results)
        elif self.margin_size is not None:
            results = self.margin_size_func(results)
        else:
            raise AttributeError('lacking variable crop_size or margin_size')

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={};margin_size={})'.format(self.crop_size, self.margin_size)
