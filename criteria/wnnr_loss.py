
import torch
import torch.nn as nn


class WeightedNuclearNormLoss(nn.Module):

    def __init__(self, num_singular_values=18):
        super().__init__()

        # Non-increasing, non-negative weights
        weights = torch.linspace(
            1.0, 0.2, steps=num_singular_values
        )

        self.register_buffer("weights", weights)

    def forward(self, latent, latent_avg=None):

        # Center the latent representation
        if latent_avg is not None:
            latent = latent - latent_avg

        # Calculate singular values
        singular_values = torch.linalg.svdvals(latent)

        weights = self.weights.to(
            device=latent.device,
            dtype=latent.dtype
        )

        # Weighted nuclear norm
        loss = (
            singular_values * weights.unsqueeze(0)
        ).sum(dim=1).mean()

        return loss
