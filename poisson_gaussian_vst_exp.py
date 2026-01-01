# SplineVST Expeirment on Poisson-Gaussian

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from scipy import stats
import matplotlib.pyplot as plt


@torch.jit.script
def B_batch_autograd(x, grid, k: int = 0):
    x = x.unsqueeze(2)
    grid = grid.unsqueeze(0)
    B = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
    for k_i in range(1, k + 1):
        g0 = grid[:, :, :-(k_i + 1)]
        gk = grid[:, :, k_i:-1]
        g1 = grid[:, :, 1:(-k_i)]
        gkp1 = grid[:, :, k_i + 1:]
        denom1 = gk - g0 + 1e-8
        denom2 = gkp1 - g1 + 1e-8
        B = ((x - g0) / denom1) * B[:, :, :-1] + ((gkp1 - x) / denom2) * B[:, :, 1:]
    return torch.nan_to_num(B).float()


def coef_to_curve(x_eval, grid, coef, k):
    b = B_batch_autograd(x_eval, grid, k=k)
    return torch.einsum('bik,iok->bio', b, coef)


class SplineVST(nn.Module):
    """Monotone cubic B-spline transform with variance-flatness regularizer."""

    def __init__(
        self,
        input_dim,
        num_knots=20,
        num_bins=32,
        momentum=0.1,
        min_density=1.0,
        grid_range=(-1, 1),
        k=3,
        binning_mode='auto',
        bandwidth_scale=0.75,
        min_delta_frac: float = 0.01,
        use_self_consistent_proxy: bool = False,
        self_consistent_iters: int = 1,
        self_consistent_base: str = 'identity',
        self_consistent_detach: bool = True,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.num_knots = num_knots
        self.num_bins = num_bins
        self.momentum = momentum
        self.min_density = min_density
        self.requested_bin_mode = binning_mode
        self.bandwidth_scale = bandwidth_scale
        self.min_delta_frac = min_delta_frac
        self.use_self_consistent_proxy = use_self_consistent_proxy
        self.self_consistent_iters = self_consistent_iters
        self.self_consistent_base = self_consistent_base
        self.self_consistent_detach = self_consistent_detach

        log_internal = self._log_spaced_knots(grid_range, num_knots + 1)
        internal = torch.einsum(
            'i,j->ij',
            torch.ones(input_dim),
            log_internal,
        )
        self.register_buffer('grid', self._extend_grid(internal, k))

        gk = num_knots + k
        self.c0 = nn.Parameter(torch.zeros(input_dim, 1, 1))
        self.udelta = nn.Parameter(torch.zeros(input_dim, 1, gk - 1))

        with torch.no_grad():
            lo, hi = self.grid[:, k], self.grid[:, -(k + 1)]
            ctrl = torch.stack(
                [torch.linspace(lo[i].item(), hi[i].item(), gk) for i in range(input_dim)],
                dim=0,
            )[:, None, :]
            self.c0.copy_(ctrl[..., :1])
            deltas = ctrl[..., 1:] - ctrl[..., :-1]
            self.udelta.copy_(torch.log(torch.expm1(deltas.clamp_min(1e-6))))
            target_range = ctrl[..., -1] - ctrl[..., 0]
            self.register_buffer('target_range', target_range)

        self.register_buffer('bin_centers', torch.zeros(input_dim, num_bins))
        self.register_buffer('running_var', torch.ones(input_dim, num_bins))
        self.register_buffer('initialized', torch.tensor(0.0))
        self.register_buffer('bin_sigma', torch.ones(input_dim))
        self.register_buffer('norm_shift', torch.zeros(input_dim))
        self.register_buffer('norm_scale', torch.ones(input_dim))
        self.register_buffer('use_log_proxy', torch.zeros(input_dim, dtype=torch.bool))

    @staticmethod
    def _extend_grid(grid, k):
        h = (grid[:, -1] - grid[:, 0]) / (grid.shape[1] - 1)
        h = h.unsqueeze(1)
        ext = grid
        for _ in range(k):
            ext = torch.cat([ext[:, :1] - h, ext, ext[:, -1:] + h], dim=1)
        return ext

    @staticmethod
    def _log_spaced_knots(grid_range, num_knots):
        lo, hi = grid_range
        if lo == hi:
            return torch.full((num_knots,), lo)

        def signed_log1p(x):
            return math.copysign(math.log1p(abs(x)), x)

        def signed_expm1(x):
            return math.copysign(math.expm1(abs(x)), x)

        lo_t = signed_log1p(lo)
        hi_t = signed_log1p(hi)
        t_vals = torch.linspace(lo_t, hi_t, steps=num_knots)
        return torch.tensor([signed_expm1(v.item()) for v in t_vals], dtype=torch.float32)

    def initialize(self, x_init):
        x_init = x_init.to(self.grid.device)
        self.init_bins_from_data(x_init)

    def coefficients(self):
        raw = F.softplus(self.udelta) + 1e-6
        sum_raw = raw.sum(dim=-1, keepdim=True)
        delta_unit = raw / (sum_raw + 1e-8)

        gk_minus1 = self.udelta.shape[-1]
        eps = self.min_delta_frac / gk_minus1
        delta_unit = (1.0 - eps * gk_minus1) * delta_unit + eps

        target_range = self.target_range.unsqueeze(-1)
        deltas = delta_unit * target_range
        return torch.cat([self.c0, deltas], dim=-1).cumsum(dim=-1)

    def forward(self, x):
        x = x.to(self.grid.device)
        x_norm = (x - self.norm_shift) / (self.norm_scale + 1e-8)

        grid = self.grid
        lo = grid[0, self.k].item()
        hi = grid[0, -(self.k + 1)].item()

        # Compute spline output for clamped values
        x_clamped = x_norm.clamp(lo, hi)
        y_spline = coef_to_curve(x_clamped, grid, self.coefficients(), self.k)

        # If extrapolation params are cached, use linear extrapolation outside grid
        if hasattr(self, '_extrap_lo') and self._extrap_lo is not None:
            y_extrap_lo = self._y_lo + self._slope_lo * (x_norm - lo)
            y_extrap_hi = self._y_hi + self._slope_hi * (x_norm - hi)

            # Match shapes: y_spline is [batch, dim, 1], extrapolated are [batch, dim]
            y_extrap_lo = y_extrap_lo.unsqueeze(-1)
            y_extrap_hi = y_extrap_hi.unsqueeze(-1)

            y = torch.where(
                x_norm.unsqueeze(-1) < lo,
                y_extrap_lo,
                torch.where(x_norm.unsqueeze(-1) > hi, y_extrap_hi, y_spline)
            )
        else:
            y = y_spline

        return y.squeeze(-1)

    def _decide_proxy_mode(self, x):
        if self.requested_bin_mode == 'log1p':
            return torch.ones(self.input_dim, dtype=torch.bool, device=x.device)
        if self.requested_bin_mode == 'linear':
            return torch.zeros(self.input_dim, dtype=torch.bool, device=x.device)
        mins, _ = x.min(dim=0)
        maxs, _ = x.max(dim=0)
        ranges = maxs - mins
        nonnegative = mins >= -1e-3
        high_range = ranges > 50.0
        return nonnegative & high_range

    def _apply_proxy(self, x):
        if self.requested_bin_mode == 'linear':
            return x
        if self.requested_bin_mode == 'log1p':
            return torch.log1p(x.clamp_min(0.0))
        mask = self.use_log_proxy.view(1, -1)
        x_log = torch.log1p(x.clamp_min(0.0))
        return torch.where(mask, x_log, x)

    def _baseline_proxy(self, x):
        if self.self_consistent_base == 'identity':
            return x
        if self.self_consistent_base == 'log1p':
            return torch.log1p(x.clamp_min(0.0))
        raise ValueError(f'Unknown self-consistent base: {self.self_consistent_base}')

    def _init_normalization_from_data(self, x_input):
        with torch.no_grad():
            device = x_input.device
            x_clean = torch.nan_to_num(x_input, nan=0.0, posinf=x_input.max(), neginf=x_input.min())
            q = torch.tensor([0.001, 0.99], device=device)
            qs = torch.quantile(x_clean, q, dim=0)
            q_low, q_high = qs[0], qs[1]
            grid = self.grid.to(device)
            lo = grid[0, self.k]
            hi = grid[0, -(self.k + 1)]
            target_half = 0.5 * (hi - lo)
            center = 0.5 * (q_low + q_high)
            half_span = 0.5 * (q_high - q_low)
            half_span = torch.clamp(half_span, min=1e-6)
            scale = half_span / (target_half + 1e-8)
            self.norm_shift.copy_(center)
            self.norm_scale.copy_(scale)

    def init_bins_from_data(self, x_input):
        with torch.no_grad():
            x_input = x_input.to(self.grid.device)
            self._init_normalization_from_data(x_input)
            is_log = self._decide_proxy_mode(x_input)
            self.use_log_proxy.copy_(is_log)
            device = self.grid.device
            q_steps = torch.linspace(0.01, 0.99, self.num_bins, device=device)
            z_centers_1d = torch.erfinv(2.0 * q_steps - 1.0) * math.sqrt(2.0)
            self.bin_centers.copy_(z_centers_1d.unsqueeze(0).expand(self.input_dim, -1))
            diffs = z_centers_1d[1:] - z_centers_1d[:-1]
            typical_spacing = torch.median(diffs)
            sigma_scalar = self.bandwidth_scale * typical_spacing
            self.bin_sigma.copy_(torch.full_like(self.bin_sigma, sigma_scalar))
            self.initialized.fill_(1.0)

    def _rank_gauss(self, x_proxy: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x_proxy)
        bsz, _ = x.shape
        if bsz <= 1:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6
            return (x - mean) / std
        order = torch.argsort(x, dim=0)
        ranks = torch.argsort(order, dim=0).float()
        u = (ranks + 0.5) / bsz
        eps = 1e-6
        u = u.clamp(eps, 1.0 - eps)
        z = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
        return z

    def compute_vst_loss(
        self, y: torch.Tensor, x_raw: torch.Tensor, sigma_scale: float = 1.0
    ) -> torch.Tensor:
        y = y.to(self.grid.device)
        x_raw = x_raw.to(self.grid.device)

        if self.initialized.item() == 0:
            self.init_bins_from_data(x_raw)

        _, d = y.shape
        y_expanded = y.unsqueeze(-1)

        def _stats_from_proxy(z_proxy: torch.Tensor):
            sigma = (self.bin_sigma * sigma_scale).view(1, d, 1) + 1e-8
            dists = (z_proxy.unsqueeze(-1) - self.bin_centers.unsqueeze(0)) / sigma
            w = torch.exp(-0.5 * dists**2) + 1e-12
            w = w / w.sum(dim=-1, keepdim=True)
            w_sum = w.sum(dim=0)
            valid_mask = w_sum > self.min_density
            mean_k = (w * y_expanded).sum(dim=0) / (w_sum + 1e-12)
            var_k = (w * (y_expanded - mean_k.unsqueeze(0)) ** 2).sum(dim=0) / (w_sum + 1e-12)
            return w, w_sum, valid_mask, mean_k, var_k

        if self.use_self_consistent_proxy:
            proxy = self._baseline_proxy(x_raw.detach())
            for _ in range(self.self_consistent_iters):
                z_proxy = self._rank_gauss(proxy)
                w, w_sum, valid_mask, mean_k, _ = _stats_from_proxy(z_proxy)
                mu_hat = (w * mean_k.unsqueeze(0)).sum(dim=-1)
                proxy = mu_hat.detach() if self.self_consistent_detach else mu_hat
            z_proxy = self._rank_gauss(proxy.detach() if self.self_consistent_detach else proxy)
            w, w_sum, valid_mask, mean_k, var_k = _stats_from_proxy(z_proxy)
        else:
            x_proxy = self._apply_proxy(x_raw.detach())
            z_proxy = self._rank_gauss(x_proxy)
            w, w_sum, valid_mask, mean_k, var_k = _stats_from_proxy(z_proxy)

        if self.training:
            with torch.no_grad():
                m = self.momentum
                self.running_var[valid_mask] = (1 - m) * self.running_var[valid_mask] + m * var_k[
                    valid_mask
                ]

        var_k = torch.clamp(var_k, min=1e-8)
        log_v = torch.log(var_k)
        log_v_masked = log_v.clone()
        log_v_masked[~valid_mask] = float('nan')
        n_valid = valid_mask.sum(dim=1)
        mu_d = torch.nansum(log_v_masked, dim=1) / (n_valid + 1e-8)
        diff_sq = (log_v_masked - mu_d.unsqueeze(-1)).square()
        var_d = torch.nansum(diff_sq, dim=1) / (n_valid - 1 + 1e-8)
        valid_features = n_valid >= 2
        if not valid_features.any():
            return torch.tensor(0.0, device=y.device)
        weights = (n_valid - 1).clamp_min(1.0).float()
        weights = weights[valid_features]
        var_d_valid = var_d[valid_features]
        return (weights * var_d_valid).sum() / weights.sum()

    def compute_vst_loss_oracle(
        self,
        y: torch.Tensor,
        mu_raw: torch.Tensor,
        mu_bin_centers: torch.Tensor,
        mu_bin_sigma: torch.Tensor,
        sigma_scale: float = 1.0,
    ) -> torch.Tensor:
        device = self.grid.device
        y = y.to(device)
        mu_raw = mu_raw.to(device)
        log_mu = torch.log(mu_raw.clamp_min(1e-8)).detach()
        _, d = y.shape
        sigma = (mu_bin_sigma * sigma_scale).view(1, d, 1) + 1e-8
        dists = (log_mu.unsqueeze(-1) - mu_bin_centers.unsqueeze(0)) / sigma
        w = torch.exp(-0.5 * dists**2) + 1e-12
        w = w / w.sum(dim=-1, keepdim=True)
        w_sum = w.sum(dim=0)
        valid_mask = w_sum > self.min_density
        y_expanded = y.unsqueeze(-1)
        mean_k = (w * y_expanded).sum(dim=0) / (w_sum + 1e-12)
        var_k = (w * (y_expanded - mean_k.unsqueeze(0)) ** 2).sum(dim=0) / (w_sum + 1e-12)
        var_k = torch.clamp(var_k, min=1e-8)
        log_v = torch.log(var_k)
        log_v_masked = log_v.clone()
        log_v_masked[~valid_mask] = float('nan')
        n_valid = valid_mask.sum(dim=1)
        mu_d = torch.nansum(log_v_masked, dim=1) / (n_valid + 1e-8)
        diff_sq = (log_v_masked - mu_d.unsqueeze(-1)).square()
        var_d = torch.nansum(diff_sq, dim=1) / (n_valid - 1 + 1e-8)
        valid_features = n_valid >= 2
        if not valid_features.any():
            return torch.tensor(0.0, device=device)
        weights = (n_valid - 1).clamp_min(1.0).float()
        weights = weights[valid_features]
        var_d_valid = var_d[valid_features]
        return (weights * var_d_valid).sum() / weights.sum()

    def smoothness_loss(self):
        c = self.coefficients().squeeze(1)
        d2 = c[:, :-2] - 2 * c[:, 1:-1] + c[:, 2:]
        return d2.square().mean()

    def _compute_extrapolation_params(self):
        with torch.no_grad():
          grid = self.grid
          lo = grid[0, self.k].item()
          hi = grid[0, -(self.k + 1)].item()

          eps = 1e-4
          device = grid.device
          lo_t = torch.tensor([[lo]], device=device)
          hi_t = torch.tensor([[hi]], device=device)
          lo_eps_t = torch.tensor([[lo + eps]], device=device)
          hi_eps_t = torch.tensor([[hi - eps]], device=device)

          coef = self.coefficients()
          y_lo = coef_to_curve(lo_t, grid, coef, self.k).squeeze()
          y_hi = coef_to_curve(hi_t, grid, coef, self.k).squeeze()
          y_lo_eps = coef_to_curve(lo_eps_t, grid, coef, self.k).squeeze()
          y_hi_eps = coef_to_curve(hi_eps_t, grid, coef, self.k).squeeze()

          self.register_buffer('_extrap_lo', torch.tensor(lo, device=device))
          self.register_buffer('_extrap_hi', torch.tensor(hi, device=device))
          self.register_buffer('_y_lo', y_lo)
          self.register_buffer('_y_hi', y_hi)
          self.register_buffer('_slope_lo', (y_lo_eps - y_lo) / eps)
          self.register_buffer('_slope_hi', (y_hi - y_hi_eps) / eps)


def generalized_anscombe(y: np.ndarray, sigma: float) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(y, a_min=0, a_max=None) + 3.0 / 8.0 + sigma**2)


def simulate_poisson_gaussian(
    num_samples: int, mu_range=(0.1, 100.0), sigma=0.3, rng: np.random.Generator | None = None
):
    rng = np.random.default_rng() if rng is None else rng
    log_mu = rng.uniform(np.log(mu_range[0]), np.log(mu_range[1]), size=num_samples)
    mu = np.exp(log_mu)
    poisson_part = rng.poisson(mu)
    gaussian_part = rng.normal(loc=0.0, scale=sigma, size=num_samples)
    y = poisson_part + gaussian_part
    return y.astype(np.float32), mu.astype(np.float32)


def build_mu_bins(mu: np.ndarray, num_bins: int = 30):
    log_mu = np.log(mu + 1e-8)
    quantiles = np.linspace(0.01, 0.99, num_bins)
    centers = np.quantile(log_mu, quantiles)
    diffs = np.diff(centers)
    sigma = np.median(diffs) * 0.75
    return torch.tensor(centers, dtype=torch.float32).unsqueeze(0), torch.tensor([sigma], dtype=torch.float32)


def evaluate_variance_stability(values: np.ndarray, mu: np.ndarray, num_bins: int = 20):
    log_mu = np.log(mu + 1e-8)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(log_mu, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_ids = np.digitize(log_mu, edges) - 1
    bin_vars = []
    bin_centers = []
    bin_means = []
    bin_counts = []  # Track sample sizes
    residuals = []

    for i in range(num_bins):
        mask = bin_ids == i
        if mask.sum() < 5:
            continue
        vals = values[mask]
        mean_i = np.mean(vals)
        var_i = np.var(vals, ddof=1)
        bin_means.append(mean_i)
        bin_vars.append(var_i)
        bin_centers.append(np.exp(np.mean(log_mu[mask])))
        bin_counts.append(mask.sum())
        if var_i > 0:
            residuals.append((vals - mean_i) / np.sqrt(var_i))

    bin_vars = np.array(bin_vars)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_counts = np.array(bin_counts)
    residuals_flat = np.concatenate(residuals) if len(residuals) > 0 else np.array([])

    log_vars = np.log(bin_vars + 1e-12)
    iqr = np.subtract(*np.percentile(log_vars, [75, 25]))
    dispersion = np.var(log_vars)
    rho, _ = stats.spearmanr(bin_centers, log_vars)
    cv = np.std(bin_vars) / (np.mean(bin_vars) + 1e-12)

    if residuals_flat.size > 0:
        kurt = stats.kurtosis(residuals_flat, fisher=True, bias=False)
        skew = stats.skew(residuals_flat, bias=False)
        ad_stat = stats.anderson(residuals_flat, dist='norm').statistic
    else:
        kurt = np.nan
        skew = np.nan
        ad_stat = np.nan

    signal_rho, _ = stats.spearmanr(values, mu)
    if not np.isfinite(signal_rho):
        signal_rho = np.nan

    # Fligner-Killeen test
    fligner_groups = []
    for i in range(num_bins):
        mask = bin_ids == i
        if mask.sum() >= 5:
            fligner_groups.append(values[mask])

    if len(fligner_groups) >= 2:
        fligner_stat, fligner_p = stats.fligner(*fligner_groups)
        # Compute effect size: normalize by degrees of freedom
        # This makes the statistic comparable across different sample sizes
        df = len(fligner_groups) - 1
        fligner_stat_normalized = fligner_stat / df
    else:
        fligner_stat, fligner_p = np.nan, np.nan
        fligner_stat_normalized = np.nan

    # Bartlett's test (more powerful but assumes normality)
    if len(fligner_groups) >= 2:
        try:
            bartlett_stat, bartlett_p = stats.bartlett(*fligner_groups)
        except Exception:
            bartlett_stat, bartlett_p = np.nan, np.nan
    else:
        bartlett_stat, bartlett_p = np.nan, np.nan

    # Levene's test (another robust alternative)
    if len(fligner_groups) >= 2:
        levene_stat, levene_p = stats.levene(*fligner_groups, center='median')
    else:
        levene_stat, levene_p = np.nan, np.nan

    return {
        'bin_centers': bin_centers,
        'bin_vars': bin_vars,
        'bin_means': bin_means,
        'bin_counts': bin_counts,
        'log_var_iqr': iqr,
        'log_var_dispersion': dispersion,
        'spearman_rho': rho,
        'variance_cv': cv,
        'kurtosis': kurt,
        'skewness': skew,
        'anderson_stat': ad_stat,
        'signal_spearman_rho': signal_rho,
        'fligner_stat': fligner_stat,
        'fligner_stat_normalized': fligner_stat_normalized,
        'fligner_p': fligner_p,
        'bartlett_stat': bartlett_stat,
        'bartlett_p': bartlett_p,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
    }


def convex_vst(
    hist: np.ndarray,
    hist_centers: np.ndarray,
    bin_centers: np.ndarray,
    var_ratio: np.ndarray | None = None,
    solver: str | None = None,
    maxiter: int = 500,
):
    hist = np.asarray(hist, dtype=np.float64)
    hist_centers = np.asarray(hist_centers, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)

    if hist.shape[0] != hist_centers.shape[0]:
        raise ValueError('hist rows must align with hist_centers')
    if hist.shape[1] != bin_centers.shape[0]:
        raise ValueError('hist columns must align with bin_centers')
    if hist_centers.min() < bin_centers.min() or hist_centers.max() > bin_centers.max():
        raise ValueError(
            f'hist_centers [{hist_centers.min():.2f}, {hist_centers.max():.2f}] '
            f'outside bin_centers [{bin_centers.min():.2f}, {bin_centers.max():.2f}]'
        )
    row_sums = hist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    hist = hist / row_sums
    if var_ratio is None:
        var_ratio = np.ones(hist.shape[0], dtype=np.float64)
    var_ratio = np.asarray(var_ratio, dtype=np.float64)
    num_bins = bin_centers.shape[0]
    num_hists = hist_centers.shape[0]
    num_vars = num_bins - 1
    target_sum = float(bin_centers[-1] - bin_centers[0])
    left_inds = np.searchsorted(bin_centers, hist_centers, side='right') - 1
    left_inds = np.clip(left_inds, 0, num_bins - 2)
    right_inds = left_inds + 1
    denom = bin_centers[right_inds] - bin_centers[left_inds]
    denom[denom == 0] = 1.0
    alpha = (hist_centers - bin_centers[left_inds]) / denom
    beta = 1.0 - alpha
    x = cp.Variable(num_vars, nonneg=True)
    t = cp.Variable()
    signal = cp.hstack([cp.Constant(0.0), cp.cumsum(x)])
    constraints = [cp.sum(x) == target_sum]
    for i in range(num_hists):
        li, ri = int(left_inds[i]), int(right_inds[i])
        ai, bi = float(alpha[i]), float(beta[i])
        ref_i = bi * signal[li] + ai * signal[ri]
        diff = signal - ref_i
        quad = cp.sum(cp.multiply(hist[i], cp.square(diff))) / var_ratio[i]
        constraints.append(quad <= t)
    problem = cp.Problem(cp.Minimize(t), constraints)
    for s in ([solver] if solver else ['ECOS', 'CLARABEL', 'SCS']):
        try:
            problem.solve(solver=s, max_iters=maxiter)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break
        except Exception:
            continue
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f'ConvexVST failed: {problem.status}')
    x_opt = np.asarray(x.value).flatten()
    stabilize_fn = np.concatenate(([0.0], np.cumsum(x_opt))) + bin_centers[0]
    signal_np = stabilize_fn - bin_centers[0]
    variance = np.zeros(num_hists)
    for i in range(num_hists):
        li, ri = int(left_inds[i]), int(right_inds[i])
        ref = beta[i] * signal_np[li] + alpha[i] * signal_np[ri]
        diff = signal_np - ref
        variance[i] = np.sum(hist[i] * diff**2) / var_ratio[i]
    return variance, stabilize_fn


def build_convexvst_inputs(
    y: np.ndarray,
    mu: np.ndarray,
    num_mu_bins: int = 20,
    num_y_bins: int = 60,
):
    mu_edges = np.quantile(mu, np.linspace(0.0, 1.0, num_mu_bins + 1))
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    y_min, y_max = np.min(y), np.max(y)
    y_edges = np.linspace(y_min, y_max, num_y_bins + 1)
    bin_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    hist = np.zeros((num_mu_bins, num_y_bins), dtype=np.float64)
    for i in range(num_mu_bins):
        lo, hi = mu_edges[i], mu_edges[i + 1]
        mask = (mu >= lo) & (mu < hi if i < num_mu_bins - 1 else mu <= hi)
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(y[mask], bins=y_edges, density=True)
        hist[i] = counts

    return hist, mu_centers, bin_centers


def build_convexvst_inputs_proxy_raw(
    y: np.ndarray,
    num_proxy_bins: int = 20,
    num_y_bins: int = 60,
    y_extend_frac: float = 0.1,
):
    proxy = np.asarray(y, dtype=np.float64)
    proxy_edges = np.quantile(proxy, np.linspace(0.0, 1.0, num_proxy_bins + 1))
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    y_lo = y_min - y_extend_frac * y_range
    y_hi = y_max + y_extend_frac * y_range
    y_edges = np.linspace(y_lo, y_hi, num_y_bins + 1)
    bin_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    hist = np.zeros((num_proxy_bins, num_y_bins), dtype=np.float64)
    hist_centers = np.zeros(num_proxy_bins, dtype=np.float64)
    valid_bins = []
    for i in range(num_proxy_bins):
        lo, hi = proxy_edges[i], proxy_edges[i + 1]
        mask = (proxy >= lo) & (proxy < hi if i < num_proxy_bins - 1 else proxy <= hi)
        if mask.sum() == 0:
            continue
        hist_centers[i] = np.median(y[mask])
        valid_bins.append(i)
        counts, _ = np.histogram(y[mask], bins=y_edges)
        hist[i] = counts.astype(np.float64)
    valid_bins = np.array(valid_bins)
    hist = hist[valid_bins]
    hist_centers = hist_centers[valid_bins]
    return hist, hist_centers, bin_centers


def train_spline_vst(y: torch.Tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    model = SplineVST(input_dim=1, num_knots=28, grid_range=(-8.0, 8.0)).to(device)
    model.initialize(y)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch_size = 512
    for _ in range(1200):
        idx = torch.randint(0, y.shape[0], (batch_size,), device=device)
        batch_y = y[idx].unsqueeze(1)
        optim.zero_grad()
        transformed = model(batch_y)
        loss = model.compute_vst_loss(transformed, batch_y)
        loss = loss + 1e-3 * model.smoothness_loss()
        loss.backward()
        optim.step()

    model._compute_extrapolation_params()

    with torch.no_grad():
        full_transformed = model(y.unsqueeze(1)).cpu().squeeze(1)
    return model.cpu(), full_transformed


def train_spline_vst_self_consistent(
    y: torch.Tensor,
    base: str = 'identity',
    iters: int = 1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    model = SplineVST(
        input_dim=1,
        num_knots=28,
        grid_range=(-8.0, 8.0),
        bandwidth_scale=1.0,
        use_self_consistent_proxy=True,
        self_consistent_base=base,
        self_consistent_iters=iters,
    ).to(device)
    model.initialize(y)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch_size = 512
    for _ in range(1200):
        idx = torch.randint(0, y.shape[0], (batch_size,), device=device)
        batch_y = y[idx].unsqueeze(1)
        optim.zero_grad()
        transformed = model(batch_y)
        loss = model.compute_vst_loss(transformed, batch_y)
        loss = loss + 1e-3 * model.smoothness_loss()
        loss.backward()
        optim.step()

    model._compute_extrapolation_params()

    with torch.no_grad():
        full_transformed = model(y.unsqueeze(1)).cpu().squeeze(1)
    return model.cpu(), full_transformed


def train_spline_vst_oracle(y: torch.Tensor, mu: torch.Tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    mu = mu.to(device)
    model = SplineVST(input_dim=1, num_knots=28, grid_range=(-8.0, 8.0)).to(device)
    model.initialize(y)
    mu_bin_centers, mu_bin_sigma = build_mu_bins(mu.cpu().numpy(), num_bins=30)
    mu_bin_centers = mu_bin_centers.to(device)
    mu_bin_sigma = mu_bin_sigma.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch_size = 512
    for _ in range(1200):
        idx = torch.randint(0, y.shape[0], (batch_size,), device=device)
        batch_y = y[idx].unsqueeze(1)
        batch_mu = mu[idx].unsqueeze(1)
        optim.zero_grad()
        transformed = model(batch_y)
        loss = model.compute_vst_loss_oracle(
            transformed,
            batch_mu,
            mu_bin_centers,
            mu_bin_sigma,
        )
        loss = loss + 1e-3 * model.smoothness_loss()
        loss.backward()
        optim.step()

    model._compute_extrapolation_params()

    with torch.no_grad():
        full_transformed = model(y.unsqueeze(1)).cpu().squeeze(1)
    return model.cpu(), full_transformed


def apply_convexvst(values: np.ndarray, convex_fn: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    slopes = np.diff(convex_fn) / np.diff(bin_centers)
    left_slope = slopes[0]
    right_slope = slopes[-1]
    left_val = convex_fn[0] + left_slope * (values - bin_centers[0])
    right_val = convex_fn[-1] + right_slope * (values - bin_centers[-1])
    inner = np.interp(values, bin_centers, convex_fn)
    return np.where(values < bin_centers[0], left_val, np.where(values > bin_centers[-1], right_val, inner))


def prepare_datasets(train_size=20000, test_size=10000, sigma=0.3, mu_range=(0.1, 100.0), seed=0):
    rng = np.random.default_rng(seed)
    y_train, mu_train = simulate_poisson_gaussian(train_size, mu_range=mu_range, sigma=sigma, rng=rng)
    y_test, mu_test = simulate_poisson_gaussian(test_size, mu_range=mu_range, sigma=sigma, rng=rng)
    return (y_train, mu_train), (y_test, mu_test)


def evaluate_with_bootstrap(values: np.ndarray, mu: np.ndarray, num_bins: int = 20, num_resamples: int = 200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    base_metrics = evaluate_variance_stability(values, mu, num_bins=num_bins)
    scalar_keys = [k for k, v in base_metrics.items() if not isinstance(v, np.ndarray)]
    ci = {}
    n = mu.shape[0]
    for key in scalar_keys:
        samples = []
        for _ in range(num_resamples):
            idx = rng.integers(0, n, n)
            metrics = evaluate_variance_stability(values[idx], mu[idx], num_bins=num_bins)
            val = metrics[key]
            if np.isfinite(val):
                samples.append(val)
        if len(samples) == 0:
            ci[key] = (np.nan, np.nan)
        else:
            ci[key] = tuple(np.percentile(samples, [2.5, 97.5]))
    base_metrics['ci'] = ci
    return base_metrics


def stratify_metrics(values: np.ndarray, mu: np.ndarray, num_bins: int = 10):
    edges = np.quantile(mu, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    strata = {}
    labels = ['low', 'medium', 'high']
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        mask = (mu >= lo) & (mu <= hi if i == 2 else mu < hi)
        if mask.sum() < 50:
            strata[labels[i]] = None
            continue
        strata[labels[i]] = evaluate_with_bootstrap(values[mask], mu[mask], num_bins=num_bins)
    return strata


def plot_transforms_matched_scale(
    spline_model,
    spline_self_consistent_model,
    spline_oracle_model,
    x_plot,
    sigma,
    device,
    convex_fn_raw=None,
    bin_centers_raw=None,
    save_path='transforms_matched_scale.png'
):
    """Plot transforms with all methods rescaled to match GAT's output range."""
    gat_curve = generalized_anscombe(x_plot, sigma)
    gat_min, gat_max = gat_curve.min(), gat_curve.max()

    def rescale_to_gat(curve):
        """Rescale any curve to match GAT's output range."""
        c_min, c_max = curve.min(), curve.max()
        normalized = (curve - c_min) / (c_max - c_min + 1e-8)
        return normalized * (gat_max - gat_min) + gat_min

    # SplineVST
    with torch.no_grad():
        spline_curve = (
            spline_model(torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_self_consistent_curve = (
            spline_self_consistent_model(
                torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1)
            )
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_oracle_curve = (
            spline_oracle_model(torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
    spline_rescaled = rescale_to_gat(spline_curve)
    spline_self_consistent_rescaled = rescale_to_gat(spline_self_consistent_curve)
    spline_oracle_rescaled = rescale_to_gat(spline_oracle_curve)

    # ConvexVST curves
    convex_raw_rescaled = None

    if convex_fn_raw is not None and bin_centers_raw is not None:
        convex_raw_curve = apply_convexvst(x_plot, convex_fn_raw, bin_centers_raw)
        convex_raw_rescaled = rescale_to_gat(convex_raw_curve)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, rescale_to_gat(x_plot), label='Identity', linestyle=':', alpha=0.5, color='gray')
    plt.plot(x_plot, gat_curve, label='GAT', linewidth=2)
    plt.plot(x_plot, spline_rescaled, label='SplineVST', linewidth=2, linestyle='--')
    plt.plot(
        x_plot,
        spline_self_consistent_rescaled,
        label='SplineVST (self-consistent proxy)',
        linewidth=2,
        linestyle='-.',
    )
    plt.plot(
        x_plot,
        spline_oracle_rescaled,
        label='SplineVST (oracle loss)',
        linewidth=2,
        linestyle=':',
    )

    if convex_raw_rescaled is not None:
        plt.plot(x_plot, convex_raw_rescaled, label='ConvexVST (raw)', linewidth=1.5, linestyle=':')

    plt.xlabel('Observed value y')
    plt.ylabel('T(y)')
    plt.title('Learned vs. theoretical transform')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'Saved matched-scale transform plot to {save_path}')

    return {
        'gat': gat_curve,
        'spline': spline_rescaled,
        'spline_self_consistent': spline_self_consistent_rescaled,
        'spline_oracle': spline_oracle_rescaled,
        'convex_raw': convex_raw_rescaled,
    }


def plot_derivatives_matched_scale(
    spline_model,
    spline_self_consistent_model,
    spline_oracle_model,
    x_plot,
    sigma,
    device,
    convex_fn_raw=None,
    bin_centers_raw=None,
    save_path='derivatives_matched_scale.png'
):
    """Plot transform derivatives with all methods rescaled to match GAT's output range."""

    def numeric_derivative(curve, x):
        return np.gradient(curve, x)

    def rescale_to_match(curve, reference):
        """Rescale curve to match reference's output range."""
        c_min, c_max = curve.min(), curve.max()
        r_min, r_max = reference.min(), reference.max()
        normalized = (curve - c_min) / (c_max - c_min + 1e-8)
        return normalized * (r_max - r_min) + r_min

    # Compute GAT curve and its derivative
    gat_curve = generalized_anscombe(x_plot, sigma)
    gat_deriv = numeric_derivative(gat_curve, x_plot)

    # Classical VST derivative: 1/sqrt(y + sigma^2) for Poisson-Gaussian
    eps = 1e-6
    classical_deriv = 1.0 / np.sqrt(np.clip(x_plot, a_min=eps, a_max=None) + sigma**2)

    # SplineVST
    with torch.no_grad():
        spline_curve = (
            spline_model(torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_self_consistent_curve = (
            spline_self_consistent_model(
                torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1)
            )
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_oracle_curve = (
            spline_oracle_model(torch.tensor(x_plot, dtype=torch.float32, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
    spline_rescaled = rescale_to_match(spline_curve, gat_curve)
    spline_deriv = numeric_derivative(spline_rescaled, x_plot)
    spline_self_consistent_rescaled = rescale_to_match(spline_self_consistent_curve, gat_curve)
    spline_self_consistent_deriv = numeric_derivative(spline_self_consistent_rescaled, x_plot)
    spline_oracle_rescaled = rescale_to_match(spline_oracle_curve, gat_curve)
    spline_oracle_deriv = numeric_derivative(spline_oracle_rescaled, x_plot)

    # ConvexVST curves
    convex_raw_deriv = None

    if convex_fn_raw is not None and bin_centers_raw is not None:
        convex_raw_curve = apply_convexvst(x_plot, convex_fn_raw, bin_centers_raw)
        convex_raw_rescaled = rescale_to_match(convex_raw_curve, gat_curve)
        convex_raw_deriv = numeric_derivative(convex_raw_rescaled, x_plot)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, classical_deriv, label='Classical 1/√(y+σ²)', linestyle='--', color='black', linewidth=2)
    plt.plot(x_plot, gat_deriv, label="GAT T'(y)", linewidth=2)
    plt.plot(x_plot, spline_deriv, label="SplineVST T'(y)", linewidth=2)
    plt.plot(
        x_plot,
        spline_self_consistent_deriv,
        label="SplineVST T'(y) (self-consistent proxy)",
        linewidth=2,
    )
    plt.plot(
        x_plot,
        spline_oracle_deriv,
        label="SplineVST T'(y) (oracle loss)",
        linewidth=2,
    )

    if convex_raw_deriv is not None:
        plt.plot(x_plot, convex_raw_deriv, label="ConvexVST T'(y) (raw)", linewidth=1.5, linestyle=':')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Observed value y')
    plt.ylabel("T'(y)")
    plt.title("Transform derivatives vs. classical VST condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'Saved matched-scale derivative plot to {save_path}')

    return {
        'classical': classical_deriv,
        'gat': gat_deriv,
        'spline': spline_deriv,
        'spline_self_consistent': spline_self_consistent_deriv,
        'spline_oracle': spline_oracle_deriv,
        'convex_raw': convex_raw_deriv,
    }


def plot_residuals_diagnostic(
    y_test: np.ndarray,
    mu_test: np.ndarray,
    spline_vals_test: np.ndarray,
    gat_test: np.ndarray,
    spline_self_consistent_vals_test: np.ndarray,
    spline_oracle_vals_test: np.ndarray,
    convex_raw_test: np.ndarray,
    save_path: str = 'residuals_diagnostic.png',
    method: str = 'lowess'  # 'lowess', 'linear', or 'polynomial'
):
    """
    Plot residuals vs μ for each transform to visually assess homoscedasticity.
    For a good VST, the residual spread should be constant across μ.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_fn

    sort_idx = np.argsort(mu_test)
    mu_sorted = mu_test[sort_idx]
    log_mu = np.log(mu_sorted)

    transforms = {
        'Identity': y_test[sort_idx],
        'GAT': gat_test[sort_idx],
        'SplineVST': spline_vals_test[sort_idx],
        'SplineVST (self-consistent proxy)': spline_self_consistent_vals_test[sort_idx],
        'SplineVST (oracle loss)': spline_oracle_vals_test[sort_idx],
        'ConvexVST (raw)': convex_raw_test[sort_idx],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (name, transformed) in enumerate(transforms.items()):
        ax = axes[idx]
        transformed = transformed.astype(float)

        # Compute fitted values using chosen method
        if method == 'lowess':
            fitted = lowess_fn(transformed, log_mu, frac=0.15, return_sorted=False)
        elif method == 'linear':
            slope, intercept, _, _, _ = stats.linregress(log_mu, transformed)
            fitted = slope * log_mu + intercept
        elif method == 'polynomial':
            coeffs = np.polyfit(log_mu, transformed, deg=3)
            fitted = np.polyval(coeffs, log_mu)
        else:
            raise ValueError(f"Unknown method: {method}")

        residuals = transformed - fitted

        # Subsample for plotting
        n_plot = min(2000, len(mu_sorted))
        plot_idx = np.linspace(0, len(mu_sorted) - 1, n_plot).astype(int)

        ax.scatter(mu_sorted[plot_idx], residuals[plot_idx], alpha=0.3, s=5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Compute local standard deviation using LOWESS on squared residuals
        #local_var = lowess_fn(residuals**2, log_mu, frac=0.15, return_sorted=False)
        #local_std = np.sqrt(np.maximum(local_var, 0))

        #ax.plot(mu_sorted, 2 * local_std, color='orange', linewidth=1.5, label='±2σ (local)')
        #ax.plot(mu_sorted, -2 * local_std, color='orange', linewidth=1.5)

        ax.set_xscale('log')
        ax.set_xlabel('μ (Poisson mean)')
        ax.set_ylabel('Residual')
        ax.set_title(name)
        ax.legend(loc='upper right')

    axes[5].axis('off')

    plt.suptitle(f'Residuals vs. μ ({method} regression): Constant spread indicates successful VST', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved residuals diagnostic plot to {save_path}')


def main():
    sigma = 0.3
    (y_train, mu_train), (y_test, mu_test) = prepare_datasets(train_size=20000, test_size=10000, sigma=sigma, seed=0)

    identity_test = y_test.copy()
    gat_test = generalized_anscombe(y_test, sigma=sigma)

    hist_raw, raw_proxy_centers, bin_centers_raw = build_convexvst_inputs_proxy_raw(
        y_train, num_proxy_bins=25, num_y_bins=80
    )
    _, convex_fn_raw = convex_vst(hist_raw, raw_proxy_centers, bin_centers_raw)

    spline_model, _ = train_spline_vst(torch.tensor(y_train))
    spline_self_consistent_model, _ = train_spline_vst_self_consistent(torch.tensor(y_train))
    spline_oracle_model, _ = train_spline_vst_oracle(
        torch.tensor(y_train),
        torch.tensor(mu_train),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spline_model = spline_model.to(device)
    spline_self_consistent_model = spline_self_consistent_model.to(device)
    spline_oracle_model = spline_oracle_model.to(device)
    with torch.no_grad():
        spline_vals_test = spline_model(torch.tensor(y_test, device=device).unsqueeze(1)).cpu().squeeze(1).numpy()
        spline_self_consistent_vals_test = (
            spline_self_consistent_model(torch.tensor(y_test, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_oracle_vals_test = (
            spline_oracle_model(torch.tensor(y_test, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )

    convex_raw_test = apply_convexvst(y_test, convex_fn_raw, bin_centers_raw)

    rng = np.random.default_rng(123)
    identity_metrics = evaluate_with_bootstrap(identity_test, mu_test, num_bins=20, num_resamples=200, rng=rng)
    gat_metrics = evaluate_with_bootstrap(gat_test, mu_test, num_bins=20, num_resamples=200, rng=rng)
    spline_metrics = evaluate_with_bootstrap(spline_vals_test, mu_test, num_bins=20, num_resamples=200, rng=rng)
    spline_self_consistent_metrics = evaluate_with_bootstrap(
        spline_self_consistent_vals_test,
        mu_test,
        num_bins=20,
        num_resamples=200,
        rng=rng,
    )
    spline_oracle_metrics = evaluate_with_bootstrap(
        spline_oracle_vals_test,
        mu_test,
        num_bins=20,
        num_resamples=200,
        rng=rng,
    )
    convex_raw_metrics = evaluate_with_bootstrap(
        convex_raw_test, mu_test, num_bins=20, num_resamples=200, rng=rng
    )

    identity_strata = stratify_metrics(identity_test, mu_test)
    gat_strata = stratify_metrics(gat_test, mu_test)
    spline_strata = stratify_metrics(spline_vals_test, mu_test)
    spline_self_consistent_strata = stratify_metrics(spline_self_consistent_vals_test, mu_test)
    spline_oracle_strata = stratify_metrics(spline_oracle_vals_test, mu_test)
    convex_raw_strata = stratify_metrics(convex_raw_test, mu_test)

    def print_metrics(label, metrics):
        print(f"\n{label} metrics (test set):")
        for k, v in metrics.items():
            if k == 'ci' or isinstance(v, np.ndarray):
                continue
            ci = metrics['ci'].get(k, (np.nan, np.nan))
            print(f"  {k}: {v:.4f} (95% CI {ci[0]:.4f}, {ci[1]:.4f})")

    print_metrics('Identity', identity_metrics)
    print_metrics('Generalized Anscombe', gat_metrics)
    print_metrics('SplineVST', spline_metrics)
    print_metrics('SplineVST (self-consistent proxy)', spline_self_consistent_metrics)
    print_metrics('SplineVST (oracle loss)', spline_oracle_metrics)
    print_metrics('ConvexVST (raw proxy)', convex_raw_metrics)

    def print_strata(label, strata):
        print(f"\n{label} stratified metrics:")
        for name, metrics in strata.items():
            if metrics is None:
                print(f"  {name}: insufficient samples")
                continue
            ci = metrics['ci']
            log_disp_ci = ci.get('log_var_dispersion', (np.nan, np.nan))
            var_cv_ci = ci.get('variance_cv', (np.nan, np.nan))
            print(
                f"  {name} log_var_dispersion: {metrics['log_var_dispersion']:.4f} "
                f"(CI {log_disp_ci[0]:.4f}, {log_disp_ci[1]:.4f})"
            )
            print(
                f"  {name} variance_cv: {metrics['variance_cv']:.4f} "
                f"(CI {var_cv_ci[0]:.4f}, {var_cv_ci[1]:.4f})"
            )

    print_strata('Identity', identity_strata)
    print_strata('Generalized Anscombe', gat_strata)
    print_strata('SplineVST', spline_strata)
    print_strata('SplineVST (self-consistent proxy)', spline_self_consistent_strata)
    print_strata('SplineVST (oracle loss)', spline_oracle_strata)
    print_strata('ConvexVST (raw proxy)', convex_raw_strata)

    plt.figure(figsize=(7, 5))
    plt.plot(identity_metrics['bin_centers'], identity_metrics['bin_vars'], label='Identity variance', marker='o')
    plt.plot(gat_metrics['bin_centers'], gat_metrics['bin_vars'], label='GAT variance', marker='o')
    plt.plot(spline_metrics['bin_centers'], spline_metrics['bin_vars'], label='SplineVST variance', marker='o')
    plt.plot(
        spline_self_consistent_metrics['bin_centers'],
        spline_self_consistent_metrics['bin_vars'],
        label='SplineVST variance (self-consistent proxy)',
        marker='o',
    )
    plt.plot(
        spline_oracle_metrics['bin_centers'],
        spline_oracle_metrics['bin_vars'],
        label='SplineVST variance (oracle loss)',
        marker='o',
    )
    plt.plot(
        convex_raw_metrics['bin_centers'],
        convex_raw_metrics['bin_vars'],
        label='ConvexVST variance (raw proxy)',
        marker='o',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Poisson mean')
    plt.ylabel('Transformed variance per μ bin (test)')
    plt.title('Variance stabilization on Poisson-Gaussian data (test set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('splinevst_vs_gat.png', dpi=200)
    print('\nSaved variance plot to splinevst_vs_gat.png')

    # Plot the learned / defined transforms on a common input grid
    all_y = np.concatenate([y_train, y_test])
    x_plot = np.linspace(all_y.min(), all_y.max(), 400)

    with torch.no_grad():
        spline_curve = (
            spline_model(torch.tensor(x_plot, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_self_consistent_curve = (
            spline_self_consistent_model(torch.tensor(x_plot, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )
        spline_oracle_curve = (
            spline_oracle_model(torch.tensor(x_plot, device=device).unsqueeze(1))
            .cpu()
            .squeeze(1)
            .numpy()
        )

    identity_curve = x_plot
    gat_curve = generalized_anscombe(x_plot, sigma=sigma)
    convex_raw_curve = apply_convexvst(x_plot, convex_fn_raw, bin_centers_raw)

    def normalize(curve):
      return (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)

    # Plot normalized transforms
    plt.figure(figsize=(7, 5))
    plt.plot(x_plot, normalize(identity_curve), label='Identity')
    plt.plot(x_plot, normalize(gat_curve), label='GAT')
    plt.plot(x_plot, normalize(spline_curve), label='SplineVST')
    plt.plot(x_plot, normalize(spline_self_consistent_curve), label='SplineVST (self-consistent proxy)')
    plt.plot(x_plot, normalize(spline_oracle_curve), label='SplineVST (oracle loss)')
    plt.plot(x_plot, normalize(convex_raw_curve), label='ConvexVST (raw proxy)')
    plt.xlabel('Observed value y')
    plt.ylabel('Normalized T(y)')
    plt.title('Learned / defined transforms (normalized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('splinevst_transforms_normalized.png', dpi=200)

    # Check normalization parameters
    print("norm_shift:", spline_model.norm_shift.item())
    print("norm_scale:", spline_model.norm_scale.item())

    # What input value hits the grid boundary?
    grid_hi = spline_model.grid[0, -(spline_model.k + 1)].item()
    clamp_threshold = spline_model.norm_shift.item() + grid_hi * spline_model.norm_scale.item()
    print("Values above this get clamped:", clamp_threshold)
    print("x_plot max:", x_plot.max())

    # What fraction of test data is clamped?
    frac_clamped = (y_test > clamp_threshold).mean()
    print(f"Fraction of test data clamped: {frac_clamped:.2%}")

    # Check data density across the range
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=50, density=True, alpha=0.7)
    plt.axvline(x=80, color='r', linestyle='--', label='Peak region start')
    plt.axvline(x=100, color='r', linestyle='--', label='Peak region end')
    plt.axvline(x=clamp_threshold, color='g', linestyle='--', label='Clamp threshold')
    plt.xlabel('y value')
    plt.ylabel('Density')
    plt.title('Training data distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Plot the spline's second derivative (curvature) to see where it's unstable
    coef = spline_model.coefficients().squeeze().detach().cpu().numpy()
    d2 = coef[:-2] - 2 * coef[1:-1] + coef[2:]
    plt.plot(np.abs(d2))
    plt.xlabel('Control point index')
    plt.ylabel('|Second difference|')
    plt.title('Spline curvature (should be smooth)')

    plt.tight_layout()
    plt.savefig('spline_diagnostics.png', dpi=150)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot, gat_curve, label='GAT')
    plt.plot(x_plot, spline_curve, label='SplineVST')
    plt.plot(x_plot, spline_self_consistent_curve, label='SplineVST (self-consistent proxy)')
    plt.plot(x_plot, spline_oracle_curve, label='SplineVST (oracle loss)')
    plt.xlabel('Observed value y')
    plt.ylabel('T(y)')
    plt.title('Raw transforms (unnormalized)')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Zoom into the "peak" region
    mask = (x_plot >= 60) & (x_plot <= 130)
    plt.plot(x_plot[mask], gat_curve[mask], label='GAT')
    plt.plot(x_plot[mask], spline_curve[mask], label='SplineVST')
    plt.plot(x_plot[mask], spline_self_consistent_curve[mask], label='SplineVST (self-consistent proxy)')
    plt.plot(x_plot[mask], spline_oracle_curve[mask], label='SplineVST (oracle loss)')
    plt.xlabel('Observed value y')
    plt.ylabel('T(y)')
    plt.title('Zoomed: y ∈ [60, 130]')
    plt.legend()

    plt.tight_layout()
    plt.savefig('raw_transforms.png', dpi=150)

    # Check where the max occurs
    print(f"SplineVST max value: {spline_curve.max():.4f} at y = {x_plot[np.argmax(spline_curve)]:.2f}")
    print(f"SplineVST value at y=127 (end): {spline_curve[-1]:.4f}")
    print(f"Is the curve monotonic? {np.all(np.diff(spline_curve) >= -1e-6)}")

    # After defining x_plot and training all models
    plot_transforms_matched_scale(
        spline_model=spline_model,
        spline_self_consistent_model=spline_self_consistent_model,
        spline_oracle_model=spline_oracle_model,
        x_plot=x_plot,
        sigma=sigma,
        device=device,
        convex_fn_raw=convex_fn_raw,
        bin_centers_raw=bin_centers_raw,
    )

    plot_derivatives_matched_scale(
        spline_model=spline_model,
        spline_self_consistent_model=spline_self_consistent_model,
        spline_oracle_model=spline_oracle_model,
        x_plot=x_plot,
        sigma=sigma,
        device=device,
        convex_fn_raw=convex_fn_raw,
        bin_centers_raw=bin_centers_raw,
    )

    plot_residuals_diagnostic(
        y_test=y_test,
        mu_test=mu_test,
        spline_vals_test=spline_vals_test,
        gat_test=gat_test,
        spline_self_consistent_vals_test=spline_self_consistent_vals_test,
        spline_oracle_vals_test=spline_oracle_vals_test,
        convex_raw_test=convex_raw_test,
        save_path='residuals_diagnostic.png',
        method='lowess'
    )


if __name__ == '__main__':
    main()