"""Unit tests for VideoManager.apply_auto_color().

Run from project root:
    pytest tests/test_auto_color.py -v
"""
import sys
import types
import numpy as np
import pytest

# ---------- mock heavy deps (same pattern as test_smooth_kps.py) ----------
def _noop(*a, **k):
    pass

_ort = types.ModuleType('onnxruntime')
_ort.set_default_logger_severity = _noop
_ort.SessionOptions = type('SessionOptions', (), {'__init__': lambda s: None})
_ort.InferenceSession = type('InferenceSession', (), {'__init__': lambda *a, **k: None})

_torch = types.ModuleType('torch')
_torch.float32 = 'float32'
_torch.uint8 = 'uint8'
_torch.device = lambda x: x

class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr.copy()
    def permute(self, *dims):
        return _FakeTensor(np.transpose(self._arr, dims))
    def cpu(self):
        return self
    def numpy(self):
        return self._arr
    def type(self, t):
        return self
    def to(self, *a, **k):
        return self
    @property
    def shape(self):
        return self._arr.shape

_torch.from_numpy = lambda a: _FakeTensor(a)
_torch.empty = lambda *a, **k: None
_torch.cuda = types.SimpleNamespace(
    memory_allocated=lambda: 0,
    memory_reserved=lambda: 0,
    get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
    is_available=lambda: False,
)

_tv = types.ModuleType('torchvision')
_tv.disable_beta_transforms_warning = _noop
_tv_tr_func = types.ModuleType('torchvision.transforms.functional')
_tv_tr_func.normalize = _noop
_tv_tr = types.ModuleType('torchvision.transforms')
_tv_tr.functional = _tv_tr_func
_tv_v2 = types.ModuleType('torchvision.transforms.v2')
_tv_v2.Resize = type('Resize', (), {'__init__': lambda *a, **k: None})
_tv_tr.v2 = _tv_v2
_tv_ops = types.ModuleType('torchvision.ops')
_tv_ops.nms = _noop
_tv.transforms = _tv_tr
_tv.ops = _tv_ops

_cv2 = types.ModuleType('cv2')
import cv2 as _real_cv2
_cv2.cvtColor = _real_cv2.cvtColor
_cv2.COLOR_RGB2LAB = _real_cv2.COLOR_RGB2LAB
_cv2.COLOR_LAB2RGB = _real_cv2.COLOR_LAB2RGB
_cv2.resize = _real_cv2.resize
_cv2.INTER_LANCZOS4 = _real_cv2.INTER_LANCZOS4

for mod_name, mod in [
    ('onnxruntime', _ort), ('torch', _torch),
    ('torchvision', _tv), ('torchvision.transforms', _tv_tr),
    ('torchvision.transforms.v2', _tv_v2),
    ('torchvision.transforms.functional', _tv_tr_func),
    ('torchvision.ops', _tv_ops), ('cv2', _cv2),
]:
    sys.modules[mod_name] = mod

from rope.VideoManager import VideoManager

# ---------- helpers ----------

def _rgb_tensor(r, g, b, size=512):
    """Return a fake C×H×W tensor filled with constant RGB values."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return _FakeTensor(np.transpose(arr, (2, 0, 1)))

# ---------- tests ----------

def test_auto_color_returns_same_shape():
    vm = VideoManager()
    swap = _rgb_tensor(200, 150, 120)
    orig = _rgb_tensor(180, 130, 100)
    result = vm.apply_auto_color(swap, orig, amount=100)
    assert result.shape == swap.shape

def test_auto_color_at_zero_amount_is_unchanged():
    vm = VideoManager()
    swap = _rgb_tensor(200, 150, 120)
    orig = _rgb_tensor(100, 80, 60)
    result = vm.apply_auto_color(swap, orig, amount=0)
    result_arr = result.permute(1, 2, 0).numpy()
    assert np.allclose(result_arr[:, :, 0], 200, atol=2)
    assert np.allclose(result_arr[:, :, 1], 150, atol=2)
    assert np.allclose(result_arr[:, :, 2], 120, atol=2)

def test_auto_color_at_full_amount_shifts_toward_original():
    vm = VideoManager()
    # swap is reddish (220,100,100), orig is bluish (100,100,220)
    swap = _rgb_tensor(220, 100, 100)
    orig = _rgb_tensor(100, 100, 220)
    result = vm.apply_auto_color(swap, orig, amount=100)
    result_arr = result.permute(1, 2, 0).numpy()
    # After full LAB transfer, blue channel should increase
    assert result_arr[:, :, 2].mean() > 120

def test_auto_color_output_values_in_range():
    vm = VideoManager()
    swap = _rgb_tensor(200, 150, 120)
    orig = _rgb_tensor(50, 50, 200)
    result = vm.apply_auto_color(swap, orig, amount=100)
    result_arr = result.numpy()
    assert result_arr.min() >= 0
    assert result_arr.max() <= 255
