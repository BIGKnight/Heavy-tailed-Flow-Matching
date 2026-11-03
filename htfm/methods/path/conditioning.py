"""
Model-side path conditioning module.

Supports two conditioning modes:
  C1. Global conditioning:  path features -> projection -> add to existing embedding
  C2. FiLM modulation:      path features -> (scale, shift) -> modulate intermediate features

Both modes share a single path embedding computed from the path feature vector.
"""

import torch
import torch.nn as nn


def build_path_conditioning_module(p):
    """v15.4: single PCM constructor consumed by init_model_by_parameter.

    Returns a PathConditioningModule instance, or None if conditioning is
    disabled (path_conditioning.enabled=false) or the dataset is not an
    image dataset (current PCM is only wired into image backbones).

    Resolves feature_dim via clocks.clock_feature_dim_from_cfg so the
    model-build path doesn't have to know per-clock_type mapping logic.
    """
    from bem.datasets import is_image_dataset
    pc = p.get('path_conditioning', {}) or {}
    if not pc.get('enabled', False):
        return None
    if not is_image_dataset(p['data']['dataset']):
        return None

    pc_global = pc.get('global', {})
    pc_mod = pc.get('modulation', {})
    use_global = pc_global.get('enabled', False)
    use_mod = pc_mod.get('enabled', False)
    global_mode = pc_global.get('mode', 'add_to_t')
    mode = ('both' if (use_global and use_mod)
            else 'global' if use_global else 'modulation')

    from ..clocks import clock_feature_dim_from_cfg
    feat_dim = clock_feature_dim_from_cfg(p)

    return PathConditioningModule(
        path_feature_dim=feat_dim,
        proj_dim=pc.get('proj_dim', 128),
        mode=mode,
        global_mode=global_mode,
        hidden_dim=pc.get('proj_hidden_dim', None),
        dropout=pc.get('proj_dropout', 0.0),
    )


class PathConditioningModule(nn.Module):
    """Dedicated module for injecting path embedding into the model.

    Computes a shared path embedding from the raw path feature vector, then
    provides two conditioning interfaces:
        - global_embedding():  returns a vector to be added to the time embedding
        - modulate():          applies FiLM (scale, shift) to an intermediate feature

    Args:
        path_feature_dim:  dimension of the input path feature vector (e.g., 5)
        proj_dim:          dimension of the shared path embedding (e.g., 128)
        mode:              'global' | 'modulation' | 'both'
        modulation_targets: number of distinct modulation targets (each gets its own head)
        hidden_dim:        hidden dimension for the projection MLP
        dropout:           dropout rate for the projection layers
    """

    def __init__(
        self,
        path_feature_dim,
        proj_dim,
        mode='global',
        global_mode='add_to_t',
        modulation_targets=1,
        hidden_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.path_feature_dim = path_feature_dim
        self.proj_dim = proj_dim
        self.mode = mode
        self.global_mode = global_mode
        self.modulation_targets = modulation_targets

        if hidden_dim is None:
            hidden_dim = max(proj_dim, path_feature_dim * 2)

        self.act = nn.SiLU(inplace=False)

        # Shared projection: path_feature_dim -> proj_dim
        self.path_proj = nn.Sequential(
            nn.Linear(path_feature_dim, hidden_dim),
            self.act,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, proj_dim),
            self.act,
        )

        # Global conditioning head (projects to target embedding dim)
        # The target dim is set later via set_global_target_dim()
        self._global_head = None
        self._block_head = None
        self._global_target_dim = None

        # FiLM modulation heads: one (scale, shift) generator per target
        self._modulation_heads = nn.ModuleDict()

    def set_global_target_dim(self, target_dim):
        """Initialize the global conditioning head to project to target_dim.

        Called after the model's embedding dimension is known (e.g., time_emb_size
        for MLPModel or time_embed_dim for UNet).
        """
        self._global_target_dim = target_dim
        if self.global_mode == 'add_to_t':
            self._global_head = nn.Sequential(
                nn.Linear(self.proj_dim, target_dim),
                self.act,
            )
            self._block_head = None
        elif self.global_mode in ('independent', 'concat'):
            self._block_head = nn.Sequential(
                nn.Linear(self.proj_dim, target_dim),
                self.act,
            )
            self._global_head = None
        else:
            raise ValueError(f"Unknown global_mode '{self.global_mode}'")

    def add_modulation_target(self, name, feature_dim):
        """Register a FiLM modulation target with given feature dimension.

        Creates a head that maps path_embedding -> (scale, shift) of size feature_dim.
        Zero-initialized so modulation starts as identity (scale=0, shift=0).
        """
        linear = nn.Linear(self.proj_dim, feature_dim * 2)
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        self._modulation_heads[name] = linear

    def compute_path_embedding(self, path_features):
        """Compute the shared path embedding from raw path features.

        Args:
            path_features: [batch, path_feature_dim] tensor

        Returns:
            [batch, proj_dim] tensor
        """
        return self.path_proj(path_features)

    def global_embedding(self, path_embedding):
        """Compute the global conditioning vector from the path embedding.

        Args:
            path_embedding: [batch, proj_dim] tensor (from compute_path_embedding)

        Returns:
            [batch, global_target_dim] tensor, to be added to the time embedding
        """
        assert self._global_head is not None, \
            "Call set_global_target_dim() before using global_embedding()"
        return self._global_head(path_embedding)

    def block_path_embedding(self, path_embedding):
        assert self._block_head is not None, \
            "Call set_global_target_dim() before using block_path_embedding()"
        return self._block_head(path_embedding)

    def modulate(self, path_embedding, features, target_name):
        """Apply FiLM modulation to intermediate features.

        Computes scale and shift from the path embedding and applies:
            features_out = features * (1 + scale) + shift

        Args:
            path_embedding: [batch, proj_dim] tensor
            features:       [batch, feature_dim, ...] tensor (intermediate model features)
            target_name:    name of the modulation target (registered via add_modulation_target)

        Returns:
            [batch, feature_dim, ...] tensor of modulated features
        """
        assert target_name in self._modulation_heads, \
            f"Modulation target '{target_name}' not registered"

        head = self._modulation_heads[target_name]
        params = head(path_embedding)  # [batch, feature_dim * 2]
        feature_dim = params.shape[-1] // 2
        raw_scale = params[..., :feature_dim]
        shift = params[..., feature_dim:]
        scale = torch.tanh(raw_scale)

        while scale.dim() < features.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return features * (1 + scale) + shift

    def forward(self, path_features):
        """Compute the path embedding (convenience wrapper).

        Args:
            path_features: [batch, path_feature_dim] tensor

        Returns:
            [batch, proj_dim] path embedding tensor
        """
        return self.compute_path_embedding(path_features)
