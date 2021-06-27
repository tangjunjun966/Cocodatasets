

构建Coco数据集的集成方法，主要集成各种数据增强方法，以下将详细说明：
pipelines :数据增强方法的集成文件夹，若有新的数据增强，将pipelines中添加
    步骤一：添加数据增强.py模块，如：    resize.py 文件
    步骤二：将添加模块添加到__init__.py中，将其导入模块字典，即可
cocodata.py:是所有数据的集成方法
build_dataset.py 是数据调用接口

至于需要使用哪些数据增强模块，可用以下方式：
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















