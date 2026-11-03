"""
Path Feature Encoder and Logsignature Backend.

Implements modular path-feature extraction for conditioning generative models
on the random clock path (t, T_t). The primary feature type is the truncated
logsignature, computed via a replaceable backend interface.

Architecture:
    PathFeatureEncoder          -- top-level encoder: path tensor -> feature vector
      |
      +-- SignatureBackend      -- abstract interface for signature-like features
      |     +-- NumericLogsignatureBackend  (default v1 backend)
      |
      +-- (future backends)     -- signatory, iisignature, analytic, etc.

For a d-dimensional path at truncation level N, the logsignature has dimension
    logsig_dim(d, N) = sum_{k=1}^{N} (1/k) * sum_{m|k} mu(k/m) * d^m
where mu is the Moebius function.  For d=2, N=3: dimension = 5.

The logsignature is computed in three phases:
    1. Truncated signature via Chen's identity (incremental tensor products)
    2. Tensor logarithm: log(S) = (S-1) - (S-1)^2/2 + (S-1)^3/3 - ...
    3. Extraction of Lyndon word coordinates from log(S)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

def _mobius(n):
    """Moebius function mu(n)."""
    if n == 1:
        return 1
    factors = []
    d, x = 2, n
    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            x //= d
            if x % d == 0:
                return 0  # squared prime factor
        d += 1
    if x > 1:
        factors.append(x)
    return (-1) ** len(factors)


def witt_dimension(d, k):
    """Dimension of the free Lie algebra of homogeneous degree k over d generators.

    Uses Witt's formula: L(d, k) = (1/k) * sum_{m|k} mu(k/m) * d^m
    """
    total = sum(_mobius(k // m) * (d ** m) for m in range(1, k + 1) if k % m == 0)
    return total // k


def logsig_dim(d, depth):
    """Total logsignature dimension for a d-dimensional path at truncation depth.

    Returns sum_{k=1}^{depth} witt_dimension(d, k).
    For d=2, depth=3: returns 5.
    """
    return sum(witt_dimension(d, k) for k in range(1, depth + 1))


def sig_dim(d, depth):
    """Total signature dimension (excluding the scalar level-0 term).

    Returns d + d^2 + ... + d^depth = d*(d^depth - 1) / (d - 1).
    For d=2, depth=3: returns 14.
    """
    if d == 1:
        return depth
    return d * (d ** depth - 1) // (d - 1)


# ---------------------------------------------------------------------------
# Lyndon word enumeration
# ---------------------------------------------------------------------------

def _enumerate_lyndon_words(d, max_length):
    """Enumerate all Lyndon words over alphabet {0, 1, ..., d-1} up to max_length.

    A Lyndon word is a string that is strictly smaller (lexicographically) than
    all of its proper suffixes.  Uses exhaustive generation with Lyndon check,
    which is correct for all d and max_length.

    Returns list of tuples, sorted by (length, lexicographic order).
    Each tuple contains integers in {0, ..., d-1}.
    """
    from itertools import product as iproduct
    words = []
    for length in range(1, max_length + 1):
        for word in iproduct(range(d), repeat=length):
            if _is_lyndon(word):
                words.append(word)
    return words


def _is_lyndon(word):
    """Check if a word is a Lyndon word (strictly smaller than all proper suffixes)."""
    n = len(word)
    for i in range(1, n):
        if word[i:] <= word:
            return False
    return True


def _word_to_flat_index(word, d):
    """Convert a multi-index word (tuple of ints) to a flat index in the d^k tensor.

    E.g., for d=2: word (0, 1) -> index 0*2 + 1 = 1
                    word (1, 0, 1) -> index 1*4 + 0*2 + 1 = 5
    """
    idx = 0
    for c in word:
        idx = idx * d + c
    return idx


# Cache Lyndon words per (d, depth) to avoid recomputation
_lyndon_cache = {}


def get_lyndon_words(d, depth):
    """Get cached Lyndon words for given alphabet size and max depth."""
    key = (d, depth)
    if key not in _lyndon_cache:
        _lyndon_cache[key] = _enumerate_lyndon_words(d, depth)
    return _lyndon_cache[key]


def get_lyndon_indices(d, depth):
    """Get flat tensor indices for Lyndon words at each level.

    Returns a dict mapping level k -> list of flat indices into the d^k tensor,
    one per Lyndon word of length k.
    """
    words = get_lyndon_words(d, depth)
    indices = {}
    for w in words:
        k = len(w)
        if k not in indices:
            indices[k] = []
        indices[k].append(_word_to_flat_index(w, d))
    return indices


# ---------------------------------------------------------------------------
# Tensor algebra operations (for batched flat-vector representation)
# ---------------------------------------------------------------------------

def _tensor_product_levels(left, right, batch_size, d):
    """Tensor product of two flat-vector tensor levels.

    left:  [batch, d^j]  (level j)
    right: [batch, d^m]  (level m)
    Returns: [batch, d^(j+m)]  (level j+m)

    This is the outer product reshaped to a flat vector:
        result[..., i*d^m + k] = left[..., i] * right[..., k]
    """
    # left: [batch, L], right: [batch, R] -> [batch, L, R] -> [batch, L*R]
    return (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(batch_size, -1)


# ---------------------------------------------------------------------------
# Core computation: Signature via Chen's identity
# ---------------------------------------------------------------------------

def compute_signature(path, depth):
    """Compute the truncated signature of a piecewise-linear path via Chen's identity.

    Args:
        path: [batch, seq_len, d] tensor of path points
        depth: truncation level (e.g., 3)

    Returns:
        List of tensors [sig_1, sig_2, ..., sig_depth] where sig_k has shape
        [batch, d^k].  Level 0 (scalar = 1) is omitted.

    Algorithm:
        For each path increment dx, compute exp(dx) in the truncated tensor
        algebra and multiply into the running signature via Chen's identity:
            S <- S * exp(dx)
        where the multiplication is the tensor shuffle product truncated at
        the given depth.

    Complexity: O(seq_len * depth^2 * max(d^k for k <= depth))
    """
    batch, seq_len, d = path.shape
    device = path.device
    dtype = path.dtype

    # Path increments: [batch, seq_len - 1, d]
    dx = path[:, 1:, :] - path[:, :-1, :]
    num_segments = dx.shape[1]

    # Initialize signature to identity: S_0 = 1, S_k = 0 for k >= 1
    sig = [torch.zeros(batch, d ** (k + 1), device=device, dtype=dtype)
           for k in range(depth)]

    for n in range(num_segments):
        inc = dx[:, n, :]  # [batch, d]

        # Compute exp(inc) truncated at depth: exp_k = inc^{otimes k} / k!
        exp_levels = []
        power = inc  # inc^{otimes 1} = inc, shape [batch, d]
        for k in range(depth):
            exp_levels.append(power / math.factorial(k + 1))
            if k + 1 < depth:
                # power_{k+1} = power_k otimes inc
                power = _tensor_product_levels(power, inc, batch, d)

        # Chen's identity: S_new = S * exp(inc)
        # (S * E)_k = sum_{j=0}^{k} S_j otimes E_{k-j}
        # where S_0 = 1 (identity), E_0 = 1 (identity)
        new_sig = []
        for k in range(depth):
            # j=0 term: S_0 * E_k = 1 * E_k = exp_levels[k]
            level_k = exp_levels[k].clone()

            # j=k+1 term: S_{k+1} * E_0 = sig[k] * 1 = sig[k]
            # (Note: sig[k] stores level k+1 in 0-indexed list)
            level_k = level_k + sig[k]

            # j=1..k terms: S_j * E_{k-j}  (both nonzero)
            for j in range(1, k + 1):
                # sig[j-1] is level j, exp_levels[k-j] is level k+1-j
                # Their tensor product gives level k+1
                tp = _tensor_product_levels(sig[j - 1], exp_levels[k - j], batch, d)
                level_k = level_k + tp

            new_sig.append(level_k)

        sig = new_sig

    return sig


# ---------------------------------------------------------------------------
# Core computation: Tensor logarithm
# ---------------------------------------------------------------------------

def _tensor_multiply_truncated(a_levels, b_levels, depth, d, batch):
    """Multiply two elements in the truncated tensor algebra (both without level 0).

    a_levels, b_levels: lists of length `depth`, where entry k is [batch, d^(k+1)].
    These represent elements with a_0 = b_0 = 0 (level 0 is zero).

    Returns: list of same structure for the product, truncated at depth.

    Product formula (level 0 = 0 for both inputs):
        (a * b)_k = sum_{j=1}^{k-1} a_j otimes b_{k-j}
    (j ranges from 1 to k-1 because a_0 = b_0 = 0)
    """
    result = []
    for k in range(depth):
        # Level k+1 in the product
        level = torch.zeros(batch, d ** (k + 1), device=a_levels[0].device,
                            dtype=a_levels[0].dtype)
        # Sum over j=1..k (but j and k-j both >= 1, so j in [1, k))
        # In 0-indexed list: a_levels[j-1] is level j, b_levels[k-j-1+1-1] = b_levels[k-j] is level k+1-j
        # Wait, let's be careful:
        # Product level m (1-indexed) = sum_{j=1}^{m-1} a_j otimes b_{m-j}
        # In our 0-indexed lists: level m corresponds to index m-1
        # So result[k] (= level k+1) = sum_{j=1}^{k} a_{j} otimes b_{k+1-j}
        # = sum_{j=1}^{k} a_levels[j-1] otimes b_levels[k-j]
        for j in range(1, k + 1):
            # a_levels[j-1] is level j: [batch, d^j]
            # b_levels[k-j] is level k+1-j: [batch, d^(k+1-j)]
            # product is level k+1: [batch, d^(k+1)]
            if j - 1 < depth and k - j < depth:
                tp = _tensor_product_levels(a_levels[j - 1], b_levels[k - j], batch, d)
                level = level + tp
        result.append(level)
    return result


def compute_log_signature(sig_levels, depth, d, batch):
    """Compute log(S) in the truncated tensor algebra.

    Given signature levels S_1, ..., S_depth (with S_0 = 1 implicit),
    compute log(1 + A) = A - A^2/2 + A^3/3 - ... where A = S - 1.

    Since A_0 = 0, the product A^n has (A^n)_k = 0 for k < n, so the series
    terminates at n = depth.

    Args:
        sig_levels: list of [batch, d^k] tensors for k=1..depth
        depth: truncation level
        d: path dimension
        batch: batch size

    Returns:
        List of [batch, d^k] tensors for k=1..depth (the log signature in
        tensor algebra form).
    """
    # A = S - 1  (levels 1..depth; level 0 is zero)
    a_levels = [s.clone() for s in sig_levels]

    # Initialize result = A (first term of the log series)
    result = [s.clone() for s in a_levels]

    # Compute higher powers: A^n for n = 2, 3, ..., depth
    # and accumulate: result += (-1)^(n+1) / n * A^n
    a_power = a_levels  # A^1
    for n in range(2, depth + 1):
        # A^n = A^{n-1} * A  (in the truncated tensor algebra)
        a_power = _tensor_multiply_truncated(a_power, a_levels, depth, d, batch)
        coeff = ((-1) ** (n + 1)) / n
        for k in range(depth):
            result[k] = result[k] + coeff * a_power[k]

    return result


# ---------------------------------------------------------------------------
# Core computation: Extract logsignature from tensor log
# ---------------------------------------------------------------------------

def extract_logsignature(log_levels, depth, d, batch):
    """Extract the logsignature coordinates from the tensor logarithm.

    The logsignature in "words" mode is obtained by reading the tensor
    coordinates of log(S) at positions indexed by Lyndon words.  This works
    because the logsignature IS the projection of log(S) onto the free Lie
    algebra, and Lyndon words form a basis for which the coefficient at word w
    equals the tensor coordinate at multi-index w.

    Args:
        log_levels: list of [batch, d^k] tensors for k=1..depth
        depth: truncation level
        d: path dimension
        batch: batch size

    Returns:
        [batch, logsig_dim(d, depth)] tensor of logsignature features.
    """
    lyndon_idx = get_lyndon_indices(d, depth)
    features = []

    for k in range(1, depth + 1):
        if k in lyndon_idx:
            level_tensor = log_levels[k - 1]  # [batch, d^k]
            for flat_idx in lyndon_idx[k]:
                features.append(level_tensor[:, flat_idx])  # [batch]

    return torch.stack(features, dim=-1)  # [batch, logsig_dim]


# ---------------------------------------------------------------------------
# Combined: path -> logsignature
# ---------------------------------------------------------------------------

def compute_logsignature(path, depth):
    """Compute the truncated logsignature of a batched path.

    Full pipeline: path -> signature (Chen) -> log (tensor algebra) -> extract (Lyndon).

    Args:
        path: [batch, seq_len, d] tensor of path points.
               For the random clock conditioning, d=2 with channels (t, T_t).
        depth: truncation level (e.g., 3).

    Returns:
        [batch, logsig_dim(d, depth)] tensor of logsignature features.
        For d=2, depth=3: shape [batch, 5].
    """
    batch, seq_len, d = path.shape
    assert seq_len >= 2, "Path must have at least 2 points"

    sig_levels = compute_signature(path, depth)
    log_levels = compute_log_signature(sig_levels, depth, d, batch)
    logsig = extract_logsignature(log_levels, depth, d, batch)

    return logsig


# ---------------------------------------------------------------------------
# Signature Backend Interface
# ---------------------------------------------------------------------------

class SignatureBackend(nn.Module):
    """Abstract interface for signature/logsignature computation.

    Subclasses must implement forward(path, depth) and output_dim(d, depth).
    This interface allows swapping the backend without changing the encoder.

    Future backends may include:
        - signatory library (GPU-accelerated, C++ backend)
        - iisignature library (CPU, with autograd wrapper)
        - analytic solvers for piecewise-linear or jump paths
        - lead-lag aware backends
    """

    def forward(self, path, depth):
        """Compute path features from a batched path tensor.

        Args:
            path: [batch, seq_len, channels] tensor
            depth: truncation level

        Returns:
            [batch, feature_dim] tensor
        """
        raise NotImplementedError

    def output_dim(self, channels, depth):
        """Return the output feature dimension for given input channels and depth."""
        raise NotImplementedError


class NumericLogsignatureBackend(SignatureBackend):
    """Default v1 backend: pure PyTorch numeric logsignature.

    Computes truncated logsignature via:
        1. Signature via Chen's identity (incremental tensor products)
        2. Tensor logarithm in the truncated tensor algebra
        3. Extraction of Lyndon word coordinates

    This backend operates on generic path tensors [batch, seq_len, channels]
    and does not assume a fixed channel count.

    Limitations:
        - O(seq_len * depth^2 * channels^depth) complexity per sample
        - Numerical precision may degrade for very long paths or high depth
        - No GPU-kernel optimization (pure PyTorch ops)

    For production use with large paths or depth > 4, consider replacing with
    the signatory library backend.
    """

    def __init__(self):
        super().__init__()

    def forward(self, path, depth):
        """Compute logsignature features.

        Args:
            path: [batch, seq_len, channels] tensor
            depth: truncation level

        Returns:
            [batch, logsig_dim(channels, depth)] tensor
        """
        return compute_logsignature(path, depth)

    def output_dim(self, channels, depth):
        return logsig_dim(channels, depth)


# ---------------------------------------------------------------------------
# Path Feature Encoder (top-level module)
# ---------------------------------------------------------------------------

class PathFeatureEncoder(nn.Module):
    """Top-level encoder: discretized path -> fixed-dimensional feature vector.

    Responsibilities:
        - Construct the path tensor from time and clock inputs
        - Optionally normalize the clock channel
        - Call the signature/logsignature backend
        - Return the final path feature tensor

    The encoder is channel-agnostic: future extensions may include additional
    channels such as delta_T (local increments), lead-lag transforms, or
    jump-aware auxiliary channels.

    Config fields used:
        path_feature_type:     'logsignature' | 'none'
        logsignature_level:    truncation depth (e.g., 3)
        path_feature_input_channels: list of channel names, e.g. ['t', 'T']
        path_feature_normalization:  None | 'standardize' | 'scale_by_terminal'
        signature_backend:     'numeric' (default)
        path_discretization_steps: number of points in the discretized path

    Args:
        p: config dict (the standard `p` parameter dict)
    """

    def __init__(self, p):
        super().__init__()
        method = p['method']
        pc = p.get('path_conditioning', {})

        self.feature_type = pc.get('path_feature_type', 'none')
        self.depth = pc.get('logsignature_level', 3)
        self.input_channels = pc.get('path_feature_input_channels', ['t', 'effective_T'])
        self.normalization = pc.get('path_feature_normalization', None)
        self.n_path_steps = pc.get('path_discretization_steps', 50)
        backend_name = pc.get('signature_backend', 'numeric')
        self.device = p['device']
        self.detach = pc.get('path_feature_detach', False)

        self.n_channels = len(self.input_channels)

        # Initialize backend
        if self.feature_type == 'logsignature':
            if backend_name == 'numeric':
                self.backend = NumericLogsignatureBackend()
            else:
                raise ValueError(f"Unknown signature backend: {backend_name}")
            self._output_dim = self.backend.output_dim(self.n_channels, self.depth)
        elif self.feature_type in ('area', 'area_terminal'):
            # Scalar clock summaries: A(T) = ∫_0^1 T_s ds (the
            # source-covariance scale = level-2 signature area), optionally with the
            # terminal T_1 (≈ level-1 / m=1). No signature backend needed.
            self.backend = None
            self._output_dim = 1 if self.feature_type == 'area' else 2
        elif self.feature_type == 'none':
            self.backend = None
            self._output_dim = 0
        else:
            raise ValueError(f"Unknown path_feature_type: {self.feature_type}")

    @property
    def output_dim(self):
        """Dimension of the output feature vector."""
        return self._output_dim

    @property
    def enabled(self):
        return self.feature_type != 'none' and self._output_dim > 0

    def _normalize_clock_channel(self, ch):
        """Apply optional normalization to a clock channel."""
        if self.normalization == 'standardize':
            mean = ch.mean(dim=-1, keepdim=True)
            std = ch.std(dim=-1, keepdim=True).clamp(min=1e-8)
            return (ch - mean) / std
        elif self.normalization == 'scale_by_terminal':
            terminal = ch[:, -1:].clamp(min=1e-8)
            return ch / terminal
        return ch

    def build_path_tensor(self, t_grid, T_raw=None, T_effective=None):
        """Construct the path tensor from time grid and clock values.

        Args:
            t_grid:       [batch, n_steps] or [n_steps] tensor of time points
            T_raw:        [batch, n_steps] raw subordinator values (cumsum of A)
            T_effective:  [batch, n_steps] effective clock values (sigma-weighted)

        Returns:
            [batch, n_steps, n_channels] path tensor
        """
        # Determine batch size from whichever clock tensor is available
        ref = T_raw if T_raw is not None else T_effective
        assert ref is not None, "At least one of T_raw, T_effective must be provided"

        # Ensure t_grid is batched
        if t_grid.dim() == 1:
            t_grid = t_grid.unsqueeze(0).expand(ref.shape[0], -1)

        channels = []
        for ch_name in self.input_channels:
            if ch_name == 't':
                channels.append(t_grid)
            elif ch_name == 'T':
                # Raw subordinator: T_t = sum A_s (no sigma weighting)
                assert T_raw is not None, \
                    "Channel 'T' (raw subordinator) requires T_raw"
                channels.append(self._normalize_clock_channel(T_raw))
            elif ch_name == 'effective_T':
                # Effective clock: Lambda_t = sum sigma_s^2 * A_s
                assert T_effective is not None, \
                    "Channel 'effective_T' (effective clock) requires T_effective"
                channels.append(self._normalize_clock_channel(T_effective))
            else:
                raise ValueError(f"Unknown path channel: {ch_name}")

        # Stack: [batch, n_steps, n_channels]
        return torch.stack(channels, dim=-1)

    def forward(self, t_grid, T_raw=None, T_effective=None):
        """Compute path features from time grid and clock values.

        Args:
            t_grid:       [batch, n_steps] or [n_steps] tensor of time points
            T_raw:        [batch, n_steps] raw subordinator (cumsum of A)
            T_effective:  [batch, n_steps] effective clock (sigma-weighted)

        Returns:
            [batch, output_dim] tensor of path features, or None if disabled.
        """
        if not self.enabled:
            return None

        if self.feature_type in ('area', 'area_terminal'):
            return self._scalar_area_features(t_grid, T_raw=T_raw, T_effective=T_effective)

        path = self.build_path_tensor(t_grid, T_raw=T_raw, T_effective=T_effective)
        features = self.backend(path, self.depth)
        if self.detach:
            features = features.detach()
        return features

    def _scalar_area_features(self, t_grid, T_raw=None, T_effective=None):
        """Scalar clock summaries (ablation of the logsignature features).

        A(T) = ∫_0^1 T_s ds (the quantity that drives the source covariance
        Σ(T) = 2·A(T); a level-2 signature "area") computed by the trapezoid rule
        over the discretized clock, and — for 'area_terminal' — the terminal clock
        value T_1 (≈ the level-1 / m=1 increment). Uses the same clock channel and
        normalization as the logsignature conditioning (effective clock by default),
        so the ablation differs from HTFM only in the path-feature representation.
        """
        clock = T_effective if 'effective_T' in self.input_channels else T_raw
        assert clock is not None, \
            "area feature needs the effective clock (T_effective or T_raw)"
        clock = self._normalize_clock_channel(clock)            # [batch, n_steps]
        if t_grid.dim() == 1:
            t = t_grid.unsqueeze(0).expand(clock.shape[0], -1)
        else:
            t = t_grid
        A = torch.trapz(clock, t, dim=-1)                       # A(T) = ∫ T dt, [batch]
        feats = [A]
        if self.feature_type == 'area_terminal':
            feats.append(clock[:, -1])                          # T_1, [batch]
        out = torch.stack(feats, dim=-1)                        # [batch, 1 or 2]
        if self.detach:
            out = out.detach()
        return out

    def _grid_and_indices(self, T_total):
        """Shared header: pick discretization indices and the matching t-grid.

        All 3 legacy compute_clock_path_from_* helpers had this prefix verbatim;
        it lives here once now.
        """
        n_steps = min(self.n_path_steps, T_total)
        if n_steps < T_total:
            step_indices = torch.linspace(0, T_total - 1, n_steps).long()
        else:
            step_indices = torch.arange(T_total)
        t_grid = (step_indices.float() / max(T_total - 1, 1)).to(self.device)
        return step_indices, t_grid

    def compute_clock_path(self, state):
        """Polymorphic clock-path construction.

        Dispatches on `state` type (see clock_state.py):
          - TerminalA: constant-A approximation. T_effective = barsigma^2 * A;
                       T_raw uses the (step_index+1) approximation.
          - ASequence: per-step A sequence + DLPM Sigma recurrence.
                       T_effective from recurrence; T_raw via cumsum.

        Args:
            state: dataclass instance from clock_state.py.

        Returns:
            t_grid:      [n_path_steps] normalized time points.
            T_raw:       [batch, n_path_steps] raw subordinator.
            T_effective: [batch, n_path_steps] effective clock.
        """
        from .clock_state import TerminalA, ASequence

        if isinstance(state, TerminalA):
            T_total = len(state.barsigmas)
            step_indices, t_grid = self._grid_and_indices(T_total)
            bs_sub = state.barsigmas[step_indices].to(self.device)
            T_effective = bs_sub.unsqueeze(0) ** 2 * state.A_scalar.unsqueeze(-1)
            T_raw = (step_indices.float() + 1).to(self.device).unsqueeze(0) \
                    * state.A_scalar.unsqueeze(-1)
            return t_grid, T_raw, T_effective

        if isinstance(state, ASequence):
            A_seq = state.A_sequence
            T_total = A_seq.shape[0]
            step_indices, t_grid = self._grid_and_indices(T_total)

            A_flat = A_seq                                      # [T, batch, ...]
            while A_flat.dim() > 2:
                A_flat = A_flat[..., 0]                          # → [T, batch]

            s = state.sigmas.to(self.device)
            g = state.gammas.to(self.device)
            # Effective clock: Sigma_t = s_t^2 * A_t + g_t^2 * Sigma_{t-1}
            Sigma = s[0] ** 2 * A_flat[0]
            all_Sigma = [Sigma]
            for t in range(1, T_total):
                Sigma = s[t] ** 2 * A_flat[t] + g[t] ** 2 * Sigma
                all_Sigma.append(Sigma)
            all_Sigma = torch.stack(all_Sigma, dim=-1)           # [batch, T]
            T_effective = all_Sigma[:, step_indices]

            all_T_raw = A_flat.cumsum(dim=0).T                   # [batch, T]
            T_raw = all_T_raw[:, step_indices]
            return t_grid, T_raw, T_effective

        raise TypeError(f'compute_clock_path: unsupported state type {type(state).__name__}')

    # ------------------------------------------------------------------
    # Backward-compat wrappers (V8 dedup): existing callers may still use
    # these specific names; new code should call compute_clock_path(state)
    # directly with a TerminalA / ASequence dataclass.
    # ------------------------------------------------------------------

    def compute_clock_path_from_terminal_A(self, barsigmas, A_scalar):
        from .clock_state import TerminalA
        return self.compute_clock_path(TerminalA(A_scalar=A_scalar, barsigmas=barsigmas))

    def compute_clock_path_from_sigmas(self, sigmas, gammas, A_sequence):
        from .clock_state import ASequence
        return self.compute_clock_path(
            ASequence(sigmas=sigmas, gammas=gammas, A_sequence=A_sequence))
