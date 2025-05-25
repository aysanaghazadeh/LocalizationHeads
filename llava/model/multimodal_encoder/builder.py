import os
from .clip_encoder_future import CLIPVisionTower, CLIPVisionTowerS2

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if use_s2:
        return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
