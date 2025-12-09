# Minimal PoseC3D fine-tune config for our skeleton pickles
# Requires mmcv/mmengine/mmaction2 installed and repo cloned at the base dir...
# Adjust BASE_PATH if the mmaction2 repo is stored elsewhere.
# BASE_PATH = "mmaction2/configs/skeleton/posec3d"

_base_ = [
    "/Users/kevinwu_new/PycharmProjects/SkiAnalytics/decision_layer/mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
]

# Dataset
dataset_type = "PoseDataset"
# go up one level, where mmaction_skeleton/ lives
ann_file_train = "/Users/kevinwu_new/PycharmProjects/SkiAnalytics/decision_layer/src/mmaction_skeleton/train.pkl"
ann_file_val = "/Users/kevinwu_new/PycharmProjects/SkiAnalytics/decision_layer/src/mmaction_skeleton/val.pkl"

left_kp = [1,3,5,7,9,11,13,15]
right_kp = [2,4,6,8,10,12,14,16]

train_dataloader = dict(
    _delete_=True,  
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,          # "PoseDataset"
        ann_file=ann_file_train,    # the train.pkl path
        split=None,
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=48),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='PackActionInputs'),
        ],
    ),
)


val_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,          # "PoseDataset"
        ann_file=ann_file_val,      # your val.pkl path
        split=None,
        test_mode=True,
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='PackActionInputs'),
        ],
    ),
)



# Model: 5 classes - [clean0, late1, near-fall2, fall3, drag4]
model = dict(cls_head=dict[str, int](num_classes=5))


# training schedule (tuned for small dataset)
optim_wrapper = dict(
    _delete_=True,  # completely override base optim_wrapper
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.05,
    ),
    clip_grad=dict(
        max_norm=40,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type="CosineAnnealingLR", T_max=75, eta_min=1e-5, by_epoch=True, begin=5, end=80),
]

# override base train_cfg so we don't inherit conflicting keys
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_begin=1,
    val_interval=1,
)

val_evaluator = [dict(type="AccMetric")]
test_evaluator = val_evaluator

