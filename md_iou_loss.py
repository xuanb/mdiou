# Copyright (c) OpenMMLab. All rights reserved.
"""MD-IoU loss: Mahalanobis Discretized IoU loss for rotated object detection.

The loss couples two terms:

* a **centre term**, a Mahalanobis-style distance between the predicted and
  target centres expressed in the target's local frame and normalised by the
  target extent, so that it is scale free;
* an **IoU term**, obtained by sampling the target boundary into a polygon and
  integrating the ray-wise inner/outer radii of the two boxes, gated by a
  sigmoid of the normalised centre distance so that it only acts once the
  centre term has already pulled the boxes together.
"""
import math

import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn

from ..builder import ROTATED_LOSSES

# Temperature of the log-sum-exp relaxation of min/max.
SMOOTH_TAU = 0.8
# Lower bound on |cos| / |sin| when dividing by a ray direction.
SAMPLE_EPS = 1e-6
# Guard for the IoU ratio and its logarithm.
IOU_EPS = 1e-8

_LOG2 = math.log(2.0)


def smooth_max(a, b, tau=SMOOTH_TAU):
    """Differentiable relaxation of ``torch.maximum``.

    Uses the log-sum-exp identity
    ``max(a, b) ~= tau * log(exp(a/tau) + exp(b/tau)) - tau * log(2)``,
    which is smooth at ``a == b`` where the exact maximum has a kink.

    Args:
        a (torch.Tensor): First operand.
        b (torch.Tensor): Second operand, same shape as ``a``.
        tau (float): Temperature. Smaller values yield a tighter
            approximation but a sharper gradient. Defaults to 0.8.

    Returns:
        torch.Tensor: Element-wise smooth maximum, same shape as ``a``.
    """
    return tau * torch.logaddexp(a / tau, b / tau) - tau * _LOG2


def smooth_min(a, b, tau=SMOOTH_TAU):
    """Differentiable relaxation of ``torch.minimum``.

    See :func:`smooth_max` for details; implemented as
    ``-smooth_max(-a, -b)``.

    Args:
        a (torch.Tensor): First operand.
        b (torch.Tensor): Second operand, same shape as ``a``.
        tau (float): Temperature. Defaults to 0.8.

    Returns:
        torch.Tensor: Element-wise smooth minimum, same shape as ``a``.
    """
    return -tau * torch.logaddexp(-a / tau, -b / tau) + tau * _LOG2


_ANGLE_GRID_CACHE = {}


def _angle_grid(num_points, device, dtype):
    """Return a cached ``[num_points]`` grid spanning ``[0, 2 * pi]``.

    The grid only depends on ``(num_points, device, dtype)``, so it is built
    once and reused across calls.

    Args:
        num_points (int): Number of grid points.
        device (torch.device): Device of the returned tensor.
        dtype (torch.dtype): Dtype of the returned tensor.

    Returns:
        torch.Tensor: Angles of shape ``(num_points,)``.
    """
    key = (num_points, device, dtype)
    grid = _ANGLE_GRID_CACHE.get(key)
    if grid is None:
        grid = torch.linspace(
            0, 2 * math.pi, num_points, device=device, dtype=dtype)
        _ANGLE_GRID_CACHE[key] = grid
    return grid


def sample_box_boundary(boxes, num_points):
    """Sample ``num_points`` points along the boundary of each rotated box.

    In the box frame, the boundary point at polar angle ``phi`` sits at radius
    ``min(half_w / |cos(phi)|, half_h / |sin(phi)|)``; the samples are taken on
    a uniform angular grid rather than uniformly in arc length.

    Args:
        boxes (torch.Tensor): Boxes of shape ``(N, 5)`` in
            ``(cx, cy, w, h, angle)`` form.
        num_points (int): Number of samples per box.

    Returns:
        torch.Tensor: Boundary points of shape ``(N, num_points, 2)``. The
        last point repeats the first so that the polygon is closed.
    """
    device, dtype = boxes.device, boxes.dtype
    num_boxes = boxes.shape[0]

    cx, cy, w, h, theta = boxes.unbind(dim=1)
    half_w = w * 0.5
    half_h = h * 0.5

    # Ray directions in the box frame, rotated into the global frame.
    angles = _angle_grid(num_points, device, dtype).unsqueeze(0) \
        + theta.unsqueeze(1)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    radius = torch.min(
        half_w.unsqueeze(1) / cos_a.abs().clamp(min=SAMPLE_EPS),
        half_h.unsqueeze(1) / sin_a.abs().clamp(min=SAMPLE_EPS))

    local_x = radius * cos_a
    local_y = radius * sin_a

    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)

    points = torch.empty(num_boxes, num_points, 2, device=device, dtype=dtype)
    points[..., 0] = cx.unsqueeze(1) + local_x * cos_t - local_y * sin_t
    points[..., 1] = cy.unsqueeze(1) + local_x * sin_t + local_y * cos_t
    # The grid endpoint already coincides with the start; force it so that the
    # closing edge of the polygon is exact.
    points[:, -1] = points[:, 0]
    return points


def rotate_points(points, center, angle):
    """Express ``points`` in the local frame of a box.

    Translates by ``center`` and applies the inverse rotation ``R(angle)^T``,
    so the result is the point coordinates in the box's own axis-aligned frame.

    Args:
        points (torch.Tensor): Points of shape ``(N, P, 2)`` in the global
            frame.
        center (torch.Tensor): Box centres of shape ``(N, 2)``.
        angle (torch.Tensor): Box orientations of shape ``(N,)``.

    Returns:
        torch.Tensor: Points of shape ``(N, P, 2)`` in the box-local frame.
    """
    local = points - center.unsqueeze(1)
    cos_a = torch.cos(angle).unsqueeze(1)
    sin_a = torch.sin(angle).unsqueeze(1)

    rotated = torch.empty_like(points)
    rotated[..., 0] = local[..., 0] * cos_a + local[..., 1] * sin_a
    rotated[..., 1] = -local[..., 0] * sin_a + local[..., 1] * cos_a
    return rotated


def compute_polar_angles(points):
    """Polar angles of ``points`` with respect to their frame's origin.

    Args:
        points (torch.Tensor): Points of shape ``(N, P, 2)``.

    Returns:
        torch.Tensor: Angles in ``(-pi, pi]`` of shape ``(N, P)``.
    """
    return torch.atan2(points[..., 1], points[..., 0])


def ray_rectangle_distance(w, h, angles, eps=SAMPLE_EPS):
    """Distance from the origin to a rectangle boundary along ``angles``.

    Args:
        w (torch.Tensor): Box widths of shape ``(N, 1)``.
        h (torch.Tensor): Box heights of shape ``(N, 1)``.
        angles (torch.Tensor): Ray directions of shape ``(N, P)``, measured in
            the box's local frame.
        eps (float): Lower bound on ``|cos|`` / ``|sin|``. Defaults to 1e-6.

    Returns:
        torch.Tensor: Radial distances of shape ``(N, P)``.
    """
    half_w = w / 2.0
    half_h = h / 2.0
    t_x = half_w / torch.abs(torch.cos(angles)).clamp(min=eps)
    t_y = half_h / torch.abs(torch.sin(angles)).clamp(min=eps)
    return torch.min(t_x, t_y)


@weighted_loss
def mdiou_loss(pred,
               target,
               iou_weight=1.0,
               dist_fn='log1p',
               dynamic=True,
               num_points=36,
               beta=0.5,
               k_squared=4.0,
               sigmoid_slope=15.0,
               sigmoid_threshold=0.8,
               eps=IOU_EPS):
    """MD-IoU loss.

    Args:
        pred (torch.Tensor): Predicted boxes of shape ``(N, 5)`` in
            ``(cx, cy, w, h, angle)`` form.
        target (torch.Tensor): Target boxes, same shape and format as ``pred``.
        iou_weight (float): Weight of the IoU term relative to the centre term.
            Defaults to 1.0.
        dist_fn (str): Non-linearity applied to the centre distance; one of
            ``'log1p'`` (``log(1 + d)``), ``'exp'`` (``exp(d) - 1``) or
            ``'none'``. Defaults to ``'log1p'``.
        dynamic (bool): Whether to gate the IoU term by the normalised centre
            distance. Defaults to True.
        num_points (int): Number of boundary samples used by the IoU term.
            Defaults to 36.
        beta (float): Rate of the exponential saturation applied to the squared
            centre distance. Defaults to 0.5.
        k_squared (float): ``K^2`` of the box-to-Gaussian mapping. ``K = 2``
            (``k_squared = 4``) makes the 1-sigma ellipse tangent to the
            rectangle. Defaults to 4.0.
        sigmoid_slope (float): Slope of the IoU gate. Defaults to 15.0.
        sigmoid_threshold (float): Normalised centre distance at which the IoU
            gate equals 0.5. Defaults to 0.8.
        eps (float): Numerical guard. Defaults to 1e-8.

    Returns:
        torch.Tensor: Per-sample loss of shape ``(N,)``; negative samples are
        zero.
    """
    assert pred.shape == target.shape and pred.shape[-1] == 5, \
        'pred and target must both be (N, 5) tensors.'

    # Negative samples carry an all-zero target box and a zero sampling weight,
    # so they contribute nothing to the loss. Following the masking pattern
    # used by other rotated losses (e.g. GCL), they are excluded before the
    # O(num_points) boundary sampling.
    pos_idx = (target != 0).any(dim=1).nonzero().squeeze(1)
    loss = pred.new_zeros(pred.shape[0])
    if pos_idx.numel() == 0:
        return loss

    pred = pred[pos_idx]
    target = target[pos_idx]

    # ------------------------------------------------------------------
    # Centre term: Mahalanobis distance in the target's local frame.
    # ------------------------------------------------------------------
    dx = pred[:, 0] - target[:, 0]
    dy = pred[:, 1] - target[:, 1]
    _, _, w_t, h_t, angle_t = target.unbind(dim=1)
    cos_t = torch.cos(angle_t)
    sin_t = torch.sin(angle_t)

    # R(angle)^T maps a global offset into the target's local frame.
    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t

    # Normalising by (w / K, h / K) makes the distance scale free; k_squared
    # sets how tight the induced Gaussian ellipse is relative to the box.
    var_x = w_t**2 / k_squared + eps
    var_y = h_t**2 / k_squared + eps
    dist_sq = x_local**2 / var_x + y_local**2 / var_y
    exp_dist = 1 - torch.exp(-beta * dist_sq)

    if dist_fn == 'log1p':
        center_loss = torch.log1p(exp_dist)
    elif dist_fn == 'exp':
        center_loss = torch.exp(exp_dist) - 1
    elif dist_fn == 'none':
        center_loss = exp_dist
    else:
        raise ValueError(f"Invalid dist_fn {dist_fn!r}, expected one of "
                         "'log1p', 'exp', 'none'.")

    # ------------------------------------------------------------------
    # IoU term: boundary sampling + ray-wise fan integration.
    # ------------------------------------------------------------------
    if dynamic:
        # Detached: the gate only schedules when the IoU term is trusted, it
        # does not itself drive the regression.
        dist = torch.sqrt(dist_sq)
        gate = sigmoid_slope * (sigmoid_threshold - dist)
        dynamic_weight = torch.sigmoid(gate).detach()
    else:
        dynamic_weight = 1.0

    target_pts = sample_box_boundary(target, num_points)
    local_pts = rotate_points(target_pts, pred[:, :2], pred[:, 4])
    angles = compute_polar_angles(local_pts)

    # Radius of each box along every ray, measured in the predicted box frame.
    pred_dist = ray_rectangle_distance(pred[:, 2:3], pred[:, 3:4], angles)
    target_dist = torch.linalg.vector_norm(local_pts, dim=2)

    # Inner radius -> intersection fan, outer radius -> union fan.
    r_min = smooth_min(pred_dist, target_dist)
    r_max = smooth_max(pred_dist, target_dist)

    sin_delta = torch.sin(angles[:, :-1] - angles[:, 1:])
    intersection = (r_min[:, :-1] * r_min[:, 1:] * sin_delta).abs().sum(dim=1)
    union = (r_max[:, :-1] * r_max[:, 1:] * sin_delta).abs().sum(dim=1)

    iou = intersection / (union + eps)
    iou_loss = -torch.log(iou + eps) * dynamic_weight

    loss[pos_idx] = center_loss + iou_weight * iou_loss
    return loss


@ROTATED_LOSSES.register_module()
class MDLoss(nn.Module):
    """MD-IoU loss.

    Combines a Mahalanobis-style centre term with a sampled-boundary IoU term
    that is gated by the normalised centre distance. See :func:`mdiou_loss`
    for the formulation and the meaning of each hyper-parameter.

    Args:
        reduction (str): Reduction method, one of ``'none'``, ``'sum'``,
            ``'mean'``. Defaults to ``'mean'``.
        loss_weight (float): Global loss weight. Defaults to 1.0.
        dist_fn (str): Non-linearity of the centre term. Defaults to
            ``'log1p'``.
        dynamic (bool): Whether to gate the IoU term. Defaults to True.
        iou_weight (float): Weight of the IoU term. Defaults to 1.0.
        num_points (int): Boundary samples per box. Defaults to 36.
        beta (float): Exponential saturation rate. Defaults to 0.5.
        k_squared (float): ``K^2`` of the box-to-Gaussian mapping. Defaults
            to 4.0 (``K = 2``, 1-sigma ellipse tangent to the rectangle).
        sigmoid_slope (float): Slope of the IoU gate. Defaults to 15.0.
        sigmoid_threshold (float): Gate midpoint on the normalised centre
            distance. Defaults to 0.8.
        eps (float): Numerical guard. Defaults to 1e-8.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 dist_fn='log1p',
                 dynamic=True,
                 iou_weight=1.0,
                 num_points=36,
                 beta=0.5,
                 k_squared=4.0,
                 sigmoid_slope=15.0,
                 sigmoid_threshold=0.8,
                 eps=IOU_EPS,
                 **kwargs):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert dist_fn in ['none', 'log1p', 'exp']
        assert num_points >= 3, \
            'num_points must be at least 3 to form a polygon.'
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.dist_fn = dist_fn
        self.dynamic = dynamic
        self.iou_weight = iou_weight
        self.num_points = num_points
        self.beta = beta
        self.k_squared = k_squared
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_threshold = sigmoid_threshold
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted boxes of shape ``(N, 5)``.
            target (torch.Tensor): Target boxes of shape ``(N, 5)``.
            weight (torch.Tensor, optional): Per-sample weights of shape
                ``(N,)`` or ``(N, 5)``. Defaults to None.
            avg_factor (int, optional): Denominator used when reducing;
                defaults to the sum of ``weight``. Defaults to None.
            reduction_override (str, optional): Overrides ``self.reduction``.
                Defaults to None.

        Returns:
            torch.Tensor: The reduced loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction

        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        # Short-circuit: a batch without any positive sample contributes zero.
        has_positive = weight is None or bool(torch.any(weight > 0))
        if not has_positive and reduction != 'none':
            return pred.sum() * 0.0

        return self.loss_weight * mdiou_loss(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            iou_weight=self.iou_weight,
            dist_fn=self.dist_fn,
            dynamic=self.dynamic,
            num_points=self.num_points,
            beta=self.beta,
            k_squared=self.k_squared,
            sigmoid_slope=self.sigmoid_slope,
            sigmoid_threshold=self.sigmoid_threshold,
            eps=self.eps,
            **kwargs)
