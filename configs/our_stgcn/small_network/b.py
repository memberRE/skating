import os


clip_len = 1000
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        num_stages=5,
        inflate_stages=[2, 4],
        down_stages=[2, 4],
        graph_cfg=dict(layout='skating', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=30, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = os.environ['PWD'] + '/train3.pkl'
train_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='skating', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='skating', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    # dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='skating', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=4, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val'))

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 64
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook') ])

# runtime settings
log_level = 'INFO'
work_dir = os.environ['PWD'] + '/configs/our_stgcn/small_network/b'
