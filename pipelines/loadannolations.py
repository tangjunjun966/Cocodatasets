import os
import warnings
import pycocotools.mask as maskUtils
import cv2
import numpy as np
import six


class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True,
                 flags=-1
                 ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno
        self.flags = flags

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['ori_bboxes'] = ann_info['bboxes']
        if len(results['ori_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = results['img_info']['img_root']
            warnings.warn('Skip the image "{}" that has no valid gt bbox'.format(file_path))
            return None

        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['ori_masks'] = gt_masks
        results['gt_masks'] = gt_masks

        return results

    def _load_semantic_seg(self, results):
        seg_img_root = results['ann_info']['seg_img_root']
        if isinstance(seg_img_root, six.string_types):
            if not os.path.isfile(seg_img_root):
                raise FileNotFoundError('loading img file does not exist: {}'.format(seg_img_root))

            gt_semantic_seg = cv2.imread(results['ann_info']['seg_img_root'], flags=self.flags).squeeze()
            results['ori_semantic_seg'] = gt_semantic_seg
            results['gt_semantic_seg'] = gt_semantic_seg
        else:
            raise TypeError('"img" must be a numpy array or a img path ')
        return results

    def __call__(self, results):
        results['with_mask'] = self.with_mask
        results['with_seg'] = self.with_seg
        results['with_label'] = self.with_label
        results['with_bbox'] = self.with_bbox

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if results['mode'] == 'train':
            if self.with_label:
                results = self._load_labels(results)
            if self.with_mask:
                results = self._load_masks(results)
            if self.with_seg:
                results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str
