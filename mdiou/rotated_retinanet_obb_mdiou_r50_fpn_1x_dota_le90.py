_base_ = ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            type='MDLoss',
            fun="log1p", 
            loss_weight=1.0, 
            iou_weight=1.0,
            beta=0.5,
            num_points=36,
            dynamic=True,
    )))
