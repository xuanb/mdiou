import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn
import math
from ..builder import ROTATED_LOSSES
def smooth_max(a, b, tau=0.8):
    x = torch.stack([a / tau, b / tau], dim=0) 
    log_sum_exp = torch.logsumexp(x, dim=0) 
    return tau * log_sum_exp-tau*math.log(2)
def smooth_min(a, b, tau=0.8):
    x = -a / tau
    y = -b / tau
    stacked = torch.stack([x, y], dim=0) 
    log_sum_exp = torch.logsumexp(stacked, dim=0) 
    return -tau * log_sum_exp+tau*math.log(2)

def sample_box_boundary(boxes: torch.Tensor, num_points: int) -> torch.Tensor:

    device = boxes.device
    dtype = boxes.dtype
    N = boxes.shape[0]
    
    cx, cy, w, h, theta = boxes.unbind(dim=1)
    hw, hh = w * 0.5, h * 0.5
    
    # Pre-allocate output [N, num_points, 2]
    points = torch.empty(N, num_points, 2, device=device, dtype=dtype)
    
    # Angles with rotation offset
    angles = torch.linspace(0, 2 * torch.pi, num_points, device=device, dtype=dtype)
    rot_angles = angles.unsqueeze(0) + theta.unsqueeze(1)
    
    # Trig
    eps = 1e-6
    cos_a = torch.cos(rot_angles)
    sin_a = torch.sin(rot_angles)

    t = torch.min(torch.abs(hw.unsqueeze(1)/(cos_a+eps)),torch.abs(hh.unsqueeze(1)/(sin_a+eps)))
    
    # Local coords
    local_x = t * cos_a
    local_y = t * sin_a
    
    # Rotation (fused)
    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)
    
    # Write directly to output tensor
    points[..., 0] = cx.unsqueeze(1) + local_x * cos_t - local_y * sin_t
    points[..., 1] = cy.unsqueeze(1) + local_x * sin_t + local_y * cos_t
    
    # Enforce closure
    points[:, -1] = points[:, 0]
    
    return points

def rotate_points(center, angle, pts):
    local_pts = pts - center.unsqueeze(1) 
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    R = torch.stack([
        torch.stack([cos_a, -sin_a], dim=1),
        torch.stack([sin_a,  cos_a], dim=1)
    ], dim=2)  # -> [n, 2, 2]

    rotated = torch.bmm(R, local_pts.transpose(1, 2))  # [n, 2, 4]
    rotated = rotated.transpose(1, 2)                  # [n, 4, 2]

    return rotated
def compute_polar_angles(points):
    x = points[:, :, 0]  # [n, 4]
    y = points[:, :, 1]  # [n, 4]
    angles = torch.atan2(y, x)  # [n, 4]
    return angles


def ray_rectangle_distance(w, h, theta,num_points,eps=1e-6):

    half_w = w / 2.0  # [n, 1]
    half_h = h / 2.0  # [n, 1]
    half_w = half_w.expand(-1, num_points)  # [n, 4]
    half_h = half_h.expand(-1, num_points)  # [n, 4]

    cos_t = torch.cos(theta)  # [n, 4]
    sin_t = torch.sin(theta)  # [n, 4]
    abs_cos = torch.abs(cos_t).clamp(min=eps)
    t_x = half_w / abs_cos
    abs_sin = torch.abs(sin_t).clamp(min=eps)
    t_y = half_h / abs_sin
    distances = torch.min(t_x, t_y)
    return distances


@weighted_loss
def mdiou_loss(pred, target, iou_weight=1.0,dist_fn='log1p', dynamic=True,num_points=36,beta=0.5,eps=1e-8):
    """MD-IoU loss.
    Mahalanobis Discretized-IoU loss for rotated object detection.

    Args:
        pred(torch.Tensor): Predicted boxes of shape (N, 5) with format (cx, cy, w, h, angle)
        target(torch.Tensor): Target boxes of shape (N, 5) with same format
        iou_weight(float): Weight for IoU loss component relative to center loss
        dist_fn(str): Distance function for center loss:
            'log1p': log(1 + distance)
            'exp': exp(distance) - 1
            'none': raw distance
        dynamic(bool): Whether to apply dynamic weighting based on
            normalized center distance
        n_points(int): Number of points to sample on box boundary
            for IoU computation
        beta(float): Exponential scaling factor for center distance transformation
        eps(float): Small constant for numerical stability

    Returns:
        loss (torch.Tensor):Combined loss tensor of shape (N,)
    """
    # Extract coordinates
    x_p, y_p = pred[:, 0], pred[:, 1]
    x_t, y_t, w_t, h_t,angle_t = target[:, 0], target[:, 1],target[:, 2], target[:, 3], target[:, 4]
    
    # Transform to target local coordinates
    x_local = (x_p-x_t)*torch.cos(angle_t)+(y_p-y_t)*torch.sin(angle_t)
    y_local = -(x_p-x_t)*torch.sin(angle_t)+(y_p-y_t)*torch.cos(angle_t)

     # Center distance loss
    dist_sq = x_local**2/((w_t)**2/(4)+eps)+y_local**2/((h_t)**2/(4)+eps)
    exp_dist=1-torch.exp(-(beta*dist_sq))
    if dist_fn == 'log1p':
        xy_loss = torch.log1p(exp_dist)
    elif dist_fn == 'exp':
        xy_loss =torch.exp(exp_dist)-1
    elif dist_fn=='none':
        xy_loss=exp_dist

    dynamic_weight=1.0
    if dynamic:
        dynamic_weight=torch.sigmoid(15*(0.8-torch.sqrt(dist_sq))).detach()
    adaptive_points = False
    if adaptive_points:
        # 使用 detach 分离计算图
        with torch.no_grad():
            avg_dist = torch.mean(torch.sqrt(dist_sq))
            
            # 分段选择点数
            if avg_dist < 0.2:  # 很近
                num_points = 72
            elif avg_dist < 0.5:  # 中等
                num_points = 48
            elif avg_dist < 1:  # 较远
                num_points = 24
            else:  # 很远
                num_points = 24
    target_pts = sample_box_boundary(target, num_points)
    rotated_pts = rotate_points(pred[:,:2],pred[:,4],target_pts)
    angles = compute_polar_angles(rotated_pts)

    pred_dist = ray_rectangle_distance(pred[:,2:3],pred[:,3:4],angles,num_points)
    target_dist = torch.norm(rotated_pts, p=2, dim=2)
    
    dists_min = smooth_min(pred_dist,target_dist)
    dists_max = smooth_max(pred_dist,target_dist)
    
    sin_delta = torch.sin(angles[:, :-1] - angles[:, 1:])
    intersection = torch.abs(dists_min[:, :-1]*dists_min[:, 1:]*sin_delta).sum(dim=1)
    union = torch.abs(dists_max[:, :-1]*dists_max[:, 1:]*sin_delta).sum(dim=1)

    iou= intersection/(union+eps)
    iou_loss = -torch.log(iou+eps)*dynamic_weight


    return xy_loss+iou_weight*iou_loss

@ROTATED_LOSSES.register_module()
class MDLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 dist_fn='log1p',
                 dynamic=True,
                 iou_weight = 1.0,
                 num_points=36,
                 beta=0.5,
                 **kwargs):
        super(MDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert dist_fn in ['none','log1p','exp',]
        self.reduction = reduction
        self.dynamic = dynamic
        self.loss_weight = loss_weight
        self.dist_fn = dist_fn
        self.iou_weight = iou_weight
        self.num_points = num_points
        self.beta = beta
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        return mdiou_loss(
            pred,
            target,
            weight=weight,
            dist_fn=self.dist_fn,
            avg_factor=avg_factor,
            reduction=reduction,
            iou_weight = self.iou_weight,
            num_points=self.num_points,
            dynamic=self.dynamic,
            beta = self.beta,
            **kwargs) * self.loss_weight     
        
    