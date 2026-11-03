"""Smoke test for t-Flow HRRR-VIL integration: data + model + one train step."""
import dataclasses
import yaml
import torch

from tflow.trainer import TrainConfig
from tflow.data_hrrr import get_hrrr_vil
from tflow.model import TFlowModel
from tflow.flow import flow_matching_loss

cfg_dict = yaml.safe_load(open('configs/hrrr_vil_nu3p0.yml'))
known = {f.name for f in dataclasses.fields(TrainConfig)}
cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in known})
print('cfg: dataset=%s image_size=%s channels=%s channel_mult=%s nu=%s' %
      (cfg.dataset, cfg.image_size, cfg.channels, cfg.channel_mult, cfg.nu))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. data
ds = get_hrrr_vil(cfg.hrrr_cache, cfg.hrrr_train_years, image_size=cfg.image_size)
x0, _ = ds[0]
print('sample shape:', tuple(x0.shape), 'min/max/mean:', float(x0.min()), float(x0.max()), float(x0.mean()))

# 2. model at 128x128x1
model = TFlowModel(image_size=cfg.image_size, channels=cfg.channels,
                   channel_mult=tuple(cfg.channel_mult)).to(dev)
nparam = sum(p.numel() for p in model.parameters())
print('model params: %.1fM' % (nparam / 1e6))

# 3. one train step on a tiny batch
B = 2
x = torch.stack([ds[i][0] for i in range(B)]).to(dev)   # (B,1,128,128), already z-scored
print('batch shape:', tuple(x.shape))
loss = flow_matching_loss(model, x, cfg.nu, param=cfg.param)
loss.backward()
print('one-step loss:', float(loss.item()))
print('SMOKE OK')
