from torch.utils.data.dataset import Dataset
import os
from pycocotools.coco import COCO
import collections
from Cocodataset.pipelines import pipelines_cls
import numpy as np


class CocoDataSet(Dataset):
    def __init__(self, data_root, transforms_pipeline, mode='train'):
        self.data_root = data_root
        self.transforms = self.inite_transforms(transforms_pipeline)
        self.mode = mode
        self.json_root = self.get_json_root()
        self.coco = COCO(self.json_root)  # 输入是json文件

        self.img_infos = self.load_annotations()
        self.cat2class = self.get_catid2class()

    def get_json_root(self):
        json_root = None
        if self.mode == 'train':
            json_root = os.path.join(self.data_root, 'train.json')
        elif self.mode == 'val':
            json_root = os.path.join(self.data_root, 'val.json')
        elif self.mode == 'test':
            json_root = os.path.join(self.data_root, 'test.json')
        else:
            assert self.mode in ['train', 'val', 'test'], 'self.mode must be train or val or test,but be {} '.format(
                self.mode)
        assert json_root is not None, 'lacking json_root file'
        return json_root

    def load_annotations(self, min_size=32):
        # 获取coco json中获取图片信息与类别信息
        '''
            load_annotations(self, ann_file)函数作用:载入json文件，并使用cocotools读取信息
            self.cat_ids 得到category中的id，为列表
            self.cat2label 得到category中id对应数字标签，为字典
            self.img_ids  得到images中的id，为列表
            img_infos 返回将images的字典记录成列表保存（多了图片绝对路径img_root）
            '''

        self.cat_ids = self.coco.getCatIds()  # 得到category的id
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.getImgIds()  # 得到img的id
        img_infos = []
        for i in self.img_ids:
            img_info = self.coco.loadImgs([i])[0]
            img_info['img_root'] = os.path.join(self.data_root, img_info['file_name'])  # 增加了一个filename
            img_infos.append(img_info)

        if self.mode == 'train':  # 排除训练集中无box的图片
            valid_inds = []
            ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())  # 通过coco.anns的annotations信息，set得到id
            for i, img_info in enumerate(img_infos):
                if self.img_ids[i] not in ids_with_ann:
                    continue
                if min(img_info['width'], img_info['height']) >= min_size:  # 排除尺寸小于32的，因32 backbone后为1
                    valid_inds.append(i)
            img_infos = [img_infos[i] for i in valid_inds]  # 返回有效self.img_infos与self.proposals（有这个的时候）
        return img_infos  #

    def inite_transforms(self, transforms_pipeline):
        assert isinstance(transforms_pipeline, collections.abc.Sequence)
        transforms = []
        for transform_dict in transforms_pipeline:  # 循环列表
            if isinstance(transform_dict, dict):
                assert 'type' in transform_dict.keys(), '{} not in transforms_pipeline'.format(transform_dict)
                obj_type = transform_dict.pop('type')
                if transform_dict:
                    transform = pipelines_cls[obj_type](**transform_dict)  # 有参数类初始化
                else:
                    transform = pipelines_cls[obj_type]()  # 无参数类初始化
                transforms.append(transform)  # 类地址保存在列表中
            else:
                raise TypeError('transform must be callable or a dict')
        return transforms

    def get_ann_info(self, idx):
        """
        Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        img_info = self.img_infos[idx]
        img_id = img_info['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])  # 获取annation中id信息
        ann_info = self.coco.loadAnns(ann_ids)  # 通过annation中id信息组合所有annation信息

        gt_bboxes, gt_labels, gt_bboxes_ignore, gt_segmentation_ann = [], [], [], []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])  # 字符映射成整数
                gt_segmentation_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # seg_map_root = img_info['img_root']#img_info['file_name']
        seg_img_root = img_info[
            'img_root']  # 更改路径#########################################################################################
        # 注释中添加新内容
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_segmentation_ann,
            img_root=seg_img_root

        )

        return ann

    def get_catid2class(self):
        cat_infos = self.coco.cats
        catid2class = dict()
        for key, info in cat_infos.items():
            k, v = info['id'], info['name']
            catid2class[k] = v
        return catid2class

    def transforms_process(self, data):
        '''
        :param data: 来自results字典信息
        :return: 返回transforms_pipeline处理后的结果，并保存到data中
        '''
        for T in self.transforms:
            data = T(data)
            if data is None:
                return None
        return data

    def parse_train(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        cat2class = self.get_catid2class()
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       cat2class=cat2class,
                       mode=self.mode)
        return self.transforms_process(results)


    def parse_test(self, idx):
        img_info = self.img_infos[idx]
        cat2class = self.get_catid2class()
        results = dict(img_info=img_info,
                       cat2class=cat2class,
                       mode=self.mode)
        return self.transforms_process(results)

    def train(self):
        if self.mode=='test':
            data=self.parse_test(0)
        else:
            data=self.parse_train(0)
        return data

    def __getitem__(self, idx):
        if self.mode=='test':
            data=self.parse_test(idx)
        else:
            data=self.parse_train(idx)
        return data

    def __len__(self):
        return len(self.img_infos)



