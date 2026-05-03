"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        self.device = torch.device("cpu")
        self.gpu_initialized = False

    def to_device(self, device):
        device = torch.device(device)
        if self.gpu_initialized and self.device == device:
            return
        self.device = device
        self._min_b_t = torch.tensor(self.min_b, device=device, dtype=torch.float32)
        self._b_diff_t = torch.tensor(
            self.max_b - self.min_b, device=device, dtype=torch.float32
        )
        self._scaling_t = torch.tensor(
            self._r3_conf.coordinate_scaling, device=device, dtype=torch.float32
        )
        self.gpu_initialized = True

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if torch.is_tensor(t):
            if torch.any((t < 0) | (t > 1)):
                raise ValueError(f'Invalid t={t}')
            b_diff = (
                self._b_diff_t
                if self.gpu_initialized and self.device == t.device
                else torch.tensor(self.max_b - self.min_b, device=t.device, dtype=t.dtype)
            )
            return self.min_b + t * b_diff
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        if torch.is_tensor(t):
            return torch.sqrt(self.b_t(t))
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -0.5 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        if self.gpu_initialized:
            return torch.randn(n_samples, 3, device=self.device)
        return torch.randn(n_samples, 3)

    def marginal_b_t(self, t):
        if torch.is_tensor(t):
            min_b = (
                self._min_b_t
                if self.gpu_initialized and self.device == t.device
                else torch.tensor(self.min_b, device=t.device, dtype=t.dtype)
            )
            b_diff = (
                self._b_diff_t
                if self.gpu_initialized and self.device == t.device
                else torch.tensor(self.max_b - self.min_b, device=t.device, dtype=t.dtype)
            )
            return t * min_b + 0.5 * (t ** 2) * b_diff
        return t * self.min_b + 0.5 * (t ** 2) * (self.max_b - self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1))."""
        if not torch.is_tensor(x_t_1):
            x_t_1 = torch.as_tensor(x_t_1, dtype=torch.float32)
        if not self.gpu_initialized or self.device != x_t_1.device:
            self.to_device(x_t_1.device)
        x_t_1 = self._scale(x_t_1)
        t_t = t if torch.is_tensor(t) else torch.tensor(t, device=x_t_1.device)
        b_t = self.marginal_b_t(t_t) / num_t
        z_t_1 = torch.randn_like(x_t_1)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0))."""
        if torch.is_tensor(x_0) or torch.is_tensor(t):
            if not torch.is_tensor(x_0):
                x_0 = torch.as_tensor(x_0, dtype=torch.float32)
            if not self.gpu_initialized or self.device != x_0.device:
                self.to_device(x_0.device)
            t_t = t if torch.is_tensor(t) else torch.tensor(t, device=x_0.device)
            x_0 = self._scale(x_0)
            beta_t = self.marginal_b_t(t_t)
            exp_term = torch.exp(-0.5 * beta_t)
            std = torch.sqrt(1 - torch.exp(-beta_t))
            x_t = exp_term * x_0 + std * torch.randn_like(x_0)
            score_t = self.score(x_t, x_0, t_t, use_torch=True)
            x_t = self._unscale(x_t)
            return x_t, score_t

        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float):
        if torch.is_tensor(t):
            return 1 / torch.sqrt(self.conditional_var(t, use_torch=True))
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: torch.Tensor,
            score_t: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            sqrt_dt: torch.Tensor,
            z: torch.Tensor,
            mask: torch.Tensor=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step."""
        if not self.gpu_initialized or self.device != x_t.device:
            self.to_device(x_t.device)
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        perturb = (f_t - g_t ** 2 * score_t) * dt + g_t * sqrt_dt * (noise_scale * z)

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones(x_t.shape[:-1], device=x_t.device)
        x_t_1 = x_t - perturb
        if center:
            mask_sum = torch.sum(mask, dim=-1, keepdim=True)
            com = torch.sum(x_t_1, dim=-2) / (mask_sum + 1e-10)
            x_t_1 -= com.unsqueeze(-2)
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0)."""
        if use_torch or torch.is_tensor(t):
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch or torch.is_tensor(x_t) or torch.is_tensor(x_0) or torch.is_tensor(t):
            exp_fn = torch.exp
            use_torch = True
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-0.5 * self.marginal_b_t(t)) * x_0) / self.conditional_var(
            t, use_torch=use_torch
        )
