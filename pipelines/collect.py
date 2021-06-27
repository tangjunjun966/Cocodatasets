class Collect(object):
    """
    将散状信息整合到data中，其中img_meta 整合了在meta_keys中的信息，其余直接以字典整合到data中
    Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self, keys=None,img_meta=None):
        self.img_keys = ['img']
        self.img_meta_keys=['img_root', 'ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg', 'cat2class']
        self.keys = keys
        self.add_keys()

    def add_keys(self):
        if self.keys is not None:
            assert isinstance(self.keys, (list, tuple)), 'self.keys must be list or tuple'
            for key in self.keys:
                self.img_keys.append(key)
        self.img_keys = set(list(self.img_keys))

    def __call__(self, results):
        data = {}

        for key in self.img_keys:
            data[key] = results[key]
        for key in self.img_meta_keys:
            data[key] = results[key]
        data['img_info']=self.img_keys
        data['img_meta']=self.img_meta_keys
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(self.keys, self.meta_keys)




# class Collect(object):
#     """
#     将散状信息整合到data中，其中img_meta 整合了在meta_keys中的信息，其余直接以字典整合到data中
#     Collect data from the loader relevant to the specific task.
#     This is usually the last stage of the data loader pipeline. Typically keys
#     is set to some subset of "img", "proposals", "gt_bboxes",
#     "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
#
#     The "img_meta" item is always populated.  The contents of the "img_meta"
#     dictionary depends on "meta_keys". By default this includes:
#
#         - "img_shape": shape of the image input to the network as a tuple
#             (h, w, c).  Note that images may be zero padded on the bottom/right
#             if the batch tensor is larger than this shape.
#
#         - "scale_factor": a float indicating the preprocessing scale
#
#         - "flip": a boolean indicating if image flip transform was used
#
#         - "filename": path to the image file
#
#         - "ori_shape": original shape of the image as a tuple (h, w, c)
#
#         - "pad_shape": image shape after padding
#
#         - "img_norm_cfg": a dict of normalization information:
#             - mean - per channel mean subtraction
#             - std - per channel std divisor
#             - to_rgb - bool indicating if bgr was converted to rgb
#     """
#
#     def __init__(self, keys=None):
#         self.img_keys = ['img','img_root', 'ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg', 'cat2class']
#         self.keys = keys
#         self.add_keys()
#
#     def add_keys(self):
#         if self.keys is not None:
#             assert isinstance(self.keys, (list, tuple)), 'self.keys must be list or tuple'
#             for key in self.keys:
#                 self.img_keys.append(key)
#         self.img_keys = set(list(self.img_keys))
#
#     def __call__(self, results):
#         data = {}
#         for key in self.img_keys:
#             data[key] = results[key]
#         return data
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(self.keys, self.meta_keys)


