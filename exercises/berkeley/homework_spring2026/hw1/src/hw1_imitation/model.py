"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

import einops


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, self.chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        y = einops.rearrange(
            y, "b (t a) -> b t a", t=self.chunk_size, a=self.action_dim
        )
        return y

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # print(f"state.shape: {state.shape}, action_chunk.shape: {action_chunk.shape}")
        y = self(state)
        # print(f"y.shape: {y.shape}, action_chunk.shape: {action_chunk.shape}")
        total_loss = (y - action_chunk) ** 2
        summed_loss = total_loss.sum(dim=(1, 2))
        average_loss = summed_loss.mean()
        return average_loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### Done: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.input_dim = state_dim + chunk_size * action_dim + 1
        layers = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, self.chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        y = einops.rearrange(
            y, "b (t a) -> b t a", t=self.chunk_size, a=self.action_dim
        )
        return y

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        a_zero = torch.rand_like(action_chunk, device=action_chunk.device)
        tau = torch.rand((state.shape[0], 1, 1), device=action_chunk.device)
        a_tau = tau * action_chunk + (1 - tau) * a_zero
        a_tau_flat = einops.rearrange(a_tau, "b t a -> b (t a)")
        tau = einops.rearrange(tau, "b t a -> b (t a)")
        x_tau = torch.cat([state, a_tau_flat, tau], dim=-1)
        # print(f"x_tau.shape: {x_tau.shape}")
        y = self(x_tau)
        target = action_chunk - a_zero
        total_loss = (y - target) ** 2
        summed_loss = total_loss.sum(dim=(1, 2))
        average_loss = summed_loss.mean()
        return average_loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # create an initial random action state (for batch, horizon, action dimension)
        a_zero = torch.rand(
            (state.shape[0], self.chunk_size * self.action_dim), device=state.device
        )
        # Now need to step through 0 to 1 in num_steps:
        a_t = a_zero
        del_t = 1.0 / num_steps
        for step in range(num_steps):
            # wondered if I should use tau = (step+1)*step_size or step*step_size
            tau = torch.full(
                (state.shape[0], 1), fill_value=(step) * del_t, device=state.device
            )
            # print(f"tau: {tau}")
            # Concatenate state, action*horizon and tau (obs) -> batch_size x (obs size (5) + horizon*action_size (16) + 1)
            x = torch.cat([state, a_t, tau], dim=-1)
            velocity = self(x)
            a_t = a_t + einops.rearrange(velocity, "b t a -> b (t a)") * del_t
        # print(a_t.shape)
        a_t = einops.rearrange(
            a_t, "b (t a) -> b t a", t=self.chunk_size, a=self.action_dim
        )
        return a_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    """Build a policy based on the policy type."""

    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
