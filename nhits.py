from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]
        knots = knots.reshape(len(knots), self.out_features, -1)
        forecast = F.interpolate(knots, size=self.forecast_size, mode="linear")
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSBlock(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_layers: int,
        mlp_width: int,
        basis: nn.Module,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        input_size = int(np.ceil(input_size / n_pool_kernel_size))

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        hidden_layers = [nn.Linear(in_features=input_size, out_features=mlp_width)]
        for _ in range(mlp_layers):
            hidden_layers.append(
                nn.Linear(in_features=mlp_width, out_features=mlp_width)
            )
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_width, out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


# %% ../../nbs/models.nhits.ipynb 10
class NHITS(nn.Module):
    """NHITS

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.
    **References:**<br>
    -[Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza,
    Max Mergenthaler-Canseco, Artur Dubrawski (2023). "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting".
    Accepted at the Thirty-Seventh AAAI Conference on Artificial Intelligence.](https://arxiv.org/abs/2201.12886)
    """

    def __init__(
        self,
        h,
        input_size,
        n_stacks: int = 3,
        n_blocks: list = [1, 1, 1],
        mlp_layers: int = 3,
        mlp_width: int = 512,
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        dropout_prob_theta=0.0,
        activation="ReLU",
        layer_norm=False,
    ):
        super().__init__()

        blocks, norms = self.create_stack(
            h=h,
            input_size=input_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            mlp_layers=mlp_layers,
            mlp_width=mlp_width,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        self.norms = torch.nn.ModuleList(norms)

    def create_stack(
        self,
        h,
        input_size,
        n_stacks,
        n_blocks,
        mlp_layers,
        mlp_width,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        dropout_prob_theta,
        activation,
        layer_norm,
    ):
        assert (
            n_stacks
            == len(n_blocks)
            == len(n_pool_kernel_size)
            == len(n_freq_downsample)
        )
        self.h = h
        block_list = []
        norms = []
        for i in range(n_stacks):
            for block_id in range(n_blocks[i]):
                n_theta = input_size + max(h // n_freq_downsample[i], 1)
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=1,
                )

                block = NHITSBlock(
                    h=h,
                    input_size=input_size,
                    n_theta=n_theta,
                    mlp_layers=mlp_layers,
                    mlp_width=mlp_width,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                if layer_norm:
                    norm = nn.LayerNorm(input_size)
                else:
                    norm = nn.Identity()

                # Select type of evaluation and apply it to all layers of block
                block_list.append(block)
                norms.append(norm)

        return block_list, norms

    def forward(self, insample_y):
        # insample
        residuals = insample_y.flip(dims=(-1,))  # backcast init

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        for block, norm in zip(self.blocks, self.norms):
            backcast, block_forecast = block(
                insample_y=residuals,
            )
            residuals = norm(residuals - backcast)
            forecast = forecast + block_forecast

        return forecast.squeeze()
