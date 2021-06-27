import numpy as np
import torch


class DefaultFormatBundle(object):
    """
    Default formatting bundle.
    将其results中某些变量转化为张量，且为cpu张量 including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor
    - proposals: (1)to tensor
    - gt_bboxes: (1)to tensor
    - gt_bboxes_ignore: (1)to tensor
    - gt_labels: (1)to tensor
    - gt_masks: (1)to tensor
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,

    """

    def __init__(self, add_keys_lst=None):
        self.keys = ['gt_proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']
        self.add_keys_lst = add_keys_lst
        self.add_keys()

    def add_keys(self):
        if self.add_keys_lst is not None:
            assert isinstance(self.add_keys_lst, list), 'self.add_keys_lst must be list '
            for key in self.add_keys_lst:
                self.keys.append(key)

    def to_tensor(self, data):
        """Convert objects of various python types to :obj:`torch.Tensor`.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError('type {} cannot be converted to tensor.'.format(type(data)))

    def __call__(self, results):
        if 'img' in results:
            img = self.to_tensor(results['img'])
            results['img'] = img.permute(2, 0, 1).contiguous()
        if results['mode'] in ['train','val']:
            for key in self.keys:
                if key not in results:
                    continue
                results[key] = self.to_tensor(results[key])
            if 'gt_masks' in results:
                results['gt_masks'] = self.to_tensor(results['gt_masks'])
            if 'gt_semantic_seg' in results:
                results['gt_semantic_seg'] = self.to_tensor(results['gt_semantic_seg'][None, ...])
        return results

    def __repr__(self):
        return self.__class__.__name__
