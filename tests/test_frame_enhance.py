"""Unit tests for VideoManager.apply_realesrgan().

Run from project root:
    pytest tests/test_frame_enhance.py -v
"""
import sys
import types
import numpy as np
import pytest

# ---- mock heavy deps (same pattern as test_auto_color.py) ----
def _noop(*a, **k):
    pass

_ort = types.ModuleType('onnxruntime')
_ort.set_default_logger_severity = _noop
_ort.SessionOptions = type('SessionOptions', (), {'__init__': lambda s: None})
_ort.InferenceSession = type('InferenceSession', (), {'__init__': lambda *a, **k: None})

_torch = types.ModuleType('torch')
_torch.float32 = 'float32'
_torch.uint8 = 'uint8'
_torch.empty = lambda *a, **k: None
_torch.cuda = types.SimpleNamespace(
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

import cv2 as _real_cv2
_cv2 = types.ModuleType('cv2')
_cv2.resize = _real_cv2.resize
_cv2.INTER_LANCZOS4 = _real_cv2.INTER_LANCZOS4
_cv2.cvtColor = _real_cv2.cvtColor
_cv2.COLOR_RGB2LAB = _real_cv2.COLOR_RGB2LAB
_cv2.COLOR_LAB2RGB = _real_cv2.COLOR_LAB2RGB

for mod_name, mod in [
    ('onnxruntime', _ort), ('torch', _torch),
    ('torchvision', _tv), ('torchvision.transforms', _tv_tr),
    ('torchvision.transforms.v2', _tv_v2),
    ('torchvision.transforms.functional', _tv_tr_func),
    ('torchvision.ops', _tv_ops), ('cv2', _cv2),
]:
    sys.modules[mod_name] = mod

from rope.VideoManager import VideoManager

# ---- mock ONNX model ----
class MockESRGANModel:
    """Returns a 4x upscaled frame (filled with 0.5) for any input."""
    def get_inputs(self):
        return [types.SimpleNamespace(name='input')]
    def get_outputs(self):
        return [types.SimpleNamespace(name='output')]
    def run(self, output_names, feed):
        inp = list(feed.values())[0]  # 1×3×H×W float32
        h, w = inp.shape[2], inp.shape[3]
        return [np.full((1, 3, h * 4, w * 4), 0.5, dtype=np.float32)]


def test_realesrgan_x4_quadruples_dimensions():
    vm = VideoManager()
    vm.realesrgan_model = MockESRGANModel()
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    result = vm.apply_realesrgan(frame, mode=1)  # x4
    assert result.shape == (400, 600, 3)


def test_realesrgan_x2_doubles_dimensions():
    vm = VideoManager()
    vm.realesrgan_model = MockESRGANModel()
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    result = vm.apply_realesrgan(frame, mode=0)  # x2
    assert result.shape == (200, 300, 3)


def test_realesrgan_output_dtype_is_uint8():
    vm = VideoManager()
    vm.realesrgan_model = MockESRGANModel()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = vm.apply_realesrgan(frame, mode=1)
    assert result.dtype == np.uint8


def test_realesrgan_output_values_in_range():
    vm = VideoManager()
    vm.realesrgan_model = MockESRGANModel()
    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = vm.apply_realesrgan(frame, mode=1)
    assert result.min() >= 0
    assert result.max() <= 255
