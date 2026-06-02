import torch
import torch.nn as nn
import torch.nn.functional as F

from nanofold.train.util import rigid_align


def compute_diffusion_loss(x, x_gt, t, data_std_dev, compute_local_geometry=False):
    with torch.no_grad():
        x_gt_aligned = rigid_align(x_gt, x).detach()
    mse_loss = F.mse_loss(x, x_gt_aligned, reduction="none").mean(dim=(-2, -1), keepdim=True) / 3
    lddt_loss = compute_lddt_loss(x, x_gt_aligned)
    local_calpha_geometry_loss = (
        compute_local_calpha_geometry_loss(x, x_gt_aligned)
        if compute_local_geometry
        else x.new_zeros(())
    )
    diffusion_loss = (t**2 + data_std_dev**2) / (t + data_std_dev) ** 2 * (mse_loss) + lddt_loss
    return {
        "mse_loss": mse_loss.mean(),
        "lddt_loss": lddt_loss.mean(),
        "local_calpha_geometry_loss": local_calpha_geometry_loss,
        "diffusion_loss": diffusion_loss.mean(),
    }


def compute_lddt_loss(x, x_gt):
    dist = torch.linalg.vector_norm(x.unsqueeze(-3) - x.unsqueeze(-2), dim=-1)
    dist_gt = torch.linalg.vector_norm(x_gt.unsqueeze(-3) - x_gt.unsqueeze(-2), dim=-1)
    diff = torch.abs(dist - dist_gt)
    e = 0.25 * (
        (diff < 0.5).type(diff.dtype)
        + (diff < 1).type(diff.dtype)
        + (diff < 2).type(diff.dtype)
        + (diff < 4).type(diff.dtype)
    )
    mask = dist_gt < 15.0
    torch.diagonal(mask, dim1=-2, dim2=-1).zero_()
    lddt = torch.sum(mask * e, dim=(-2, -1), keepdim=True) / torch.sum(
        mask, dim=(-2, -1), keepdim=True
    )
    return 1 - lddt


def compute_local_calpha_geometry_loss(x, x_gt, cutoff=15.0):
    ca = extract_calpha_coords(x)
    ca_gt = extract_calpha_coords(x_gt)
    dist = torch.linalg.vector_norm(ca.unsqueeze(-3) - ca.unsqueeze(-2), dim=-1)
    dist_gt = torch.linalg.vector_norm(ca_gt.unsqueeze(-3) - ca_gt.unsqueeze(-2), dim=-1)
    mask = dist_gt < cutoff
    torch.diagonal(mask, dim1=-2, dim2=-1).zero_()
    pair_loss = F.smooth_l1_loss(dist, dist_gt, reduction="none")
    denominator = torch.sum(mask, dim=(-2, -1)).clamp_min(1)
    per_sample = torch.sum(pair_loss * mask, dim=(-2, -1)) / denominator
    return per_sample.mean()


def extract_calpha_coords(x, atoms_per_residue=3, calpha_index=1):
    if x.shape[-2] % atoms_per_residue != 0:
        raise ValueError("backbone atom dimension must be divisible by atoms_per_residue")
    residue_count = x.shape[-2] // atoms_per_residue
    return x.reshape(*x.shape[:-2], residue_count, atoms_per_residue, x.shape[-1])[
        ..., calpha_index, :
    ]


class DistogramLoss(nn.Module):
    def __init__(
        self,
        pair_embedding_size,
        num_bins,
        device,
        contact_auxiliary_loss_weight=0.0,
        contact_auxiliary_distance_cutoff=8.0,
        contact_auxiliary_min_sequence_separation=8,
    ):
        super().__init__()
        self.bins = torch.arange(2, 22, 20 / num_bins, device=device)
        self.projection = nn.Linear(pair_embedding_size, len(self.bins))
        self.contact_auxiliary_loss_weight = float(contact_auxiliary_loss_weight)
        self.contact_auxiliary_distance_cutoff = float(contact_auxiliary_distance_cutoff)
        self.contact_auxiliary_min_sequence_separation = int(contact_auxiliary_min_sequence_separation)

    def forward(self, pair_rep, coords_truth):
        logits = self.projection(pair_rep + pair_rep.transpose(-3, -2))
        distance_mat = torch.norm(coords_truth.unsqueeze(-2) - coords_truth.unsqueeze(-3), dim=-1)
        index = torch.argmin(torch.abs(distance_mat.unsqueeze(-1) - self.bins), dim=-1)
        per_pair_loss = nn.functional.cross_entropy(logits.transpose(-1, 1), index, reduction="none")
        if self.contact_auxiliary_loss_weight <= 0.0:
            return per_pair_loss.mean()
        weight = self._contact_auxiliary_weight(distance_mat)
        return (per_pair_loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _contact_auxiliary_weight(self, distance_mat):
        residue_count = distance_mat.shape[-1]
        residue_index = torch.arange(residue_count, device=distance_mat.device)
        sequence_separation = torch.abs(residue_index.unsqueeze(0) - residue_index.unsqueeze(1))
        contact_mask = (
            (distance_mat <= self.contact_auxiliary_distance_cutoff)
            & (sequence_separation >= self.contact_auxiliary_min_sequence_separation)
        )
        diagonal = torch.eye(residue_count, dtype=torch.bool, device=distance_mat.device)
        contact_mask = contact_mask & ~diagonal
        return 1.0 + self.contact_auxiliary_loss_weight * contact_mask.to(distance_mat.dtype)
