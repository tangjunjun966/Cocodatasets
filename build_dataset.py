from Cocodataset.cocodata import CocoDataSet
from torch.utils.data import DataLoader

import torch


class Build_Dataset():
    def __init__(self):
        self.mode = None

    def build_train_data(self, data_root, train_pipelines, mode='train'):
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        self.mode = mode
        return datasets_train

    def build_val_data(self, data_root, val_pipelines, mode='val'):
        datasets_val = CocoDataSet(data_root, val_pipelines, mode=mode)
        self.mode = mode
        return datasets_val

    def build_test_data(self, data_root, test_pipelines, mode='train'):
        datasets_test = CocoDataSet(data_root, test_pipelines, mode=mode)
        self.mode = mode
        return datasets_test

    def collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        num_batch = len(batch)
        img_info_keys = batch[0]['img_info']
        img_meta_keys = batch[0]['img_meta']
        data = {}
        for key in img_info_keys:
            key_lst = []
            for i in range(num_batch):
                key_lst.append(batch[i][key])
            if key == 'img':
                key_lst = torch.stack(key_lst, 0)
            data[key] = key_lst
        img_meta = {}
        for key in img_meta_keys:
            key_lst = []
            for i in range(num_batch):
                key_lst.append(batch[i][key])
            img_meta[key] = key_lst

        return [data, img_meta]

    def build_dataloader(self, datasets):

        dataloader = DataLoader(datasets,
                                batch_size=3,
                                num_workers=2,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=self.collate
                                )

        return dataloader


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(256, 260)),
    dict(type='Normalize'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])

]

train_pipeline = [
    dict(type='LoadImageFromFile'),  # 载入图像
    dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

    dict(type='RandomCropResize', crop_size=(426, 426), crop_ratio=1.1),

    # 加载数据处理模块#

    dict(type='Normalize'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # 在results中需要提取的结果
]

if __name__ == '__main__':
    root = r'C:\Users\51102\Desktop\Fasterrcnn_tj\Cocodataset\train_data'
    BD = Build_Dataset()
    dataset = BD.build_train_data(root, train_pipeline)
    dataloader = BD.build_dataloader(dataset)
    # for data in dataset:
    for data, img_meta in dataloader:
        print(data['img'].shape)
        print(img_meta)
