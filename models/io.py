"""Checkpoint loading."""

import torch

from .lms_trd import build_model


def load_model(checkpoint: str, device: str = 'cpu'):
    """Rebuild the model from the configuration stored in the checkpoint and load its weights."""
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt.get('model_cfg') if isinstance(ckpt, dict) else None)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()
