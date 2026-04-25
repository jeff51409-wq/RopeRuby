"""Unit tests for VideoManager temporal KPS smoothing.

Run from project root:
    pytest tests/test_smooth_kps.py -v
"""
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock heavy dependencies so VideoManager can be imported without GPU.
# ---------------------------------------------------------------------------
def _noop(*a, **k):
    pass


_ort = types.ModuleType('onnxruntime')
_ort.set_default_logger_severity = _noop

_torch = types.ModuleType('torch')
_torch.float32 = 'float32'
_torch.empty = lambda *a, **k: None
_torch.cuda = types.SimpleNamespace(
    memory_allocated=lambda: 0,
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

_ski_tr = types.ModuleType('skimage.transform')
_ski = types.ModuleType('skimage')
_ski.transform = _ski_tr

_pil_img = types.ModuleType('PIL.Image')
_pil_imgtk = types.ModuleType('PIL.ImageTk')
_pil = types.ModuleType('PIL')
_pil.Image = _pil_img
_pil.ImageTk = _pil_imgtk

for _name, _mod in [
    ('onnxruntime', _ort),
    ('torch', _torch),
    ('torchvision', _tv),
    ('torchvision.transforms', _tv_tr),
    ('torchvision.transforms.functional', _tv_tr_func),
    ('torchvision.transforms.v2', _tv_v2),
    ('torchvision.ops', _tv_ops),
    ('cv2', types.ModuleType('cv2')),
    ('PIL', _pil),
    ('PIL.Image', _pil_img),
    ('PIL.ImageTk', _pil_imgtk),
    ('skimage', _ski),
    ('skimage.transform', _ski_tr),
    ('tkinter', types.ModuleType('tkinter')),
    ('tkinter.ttk', types.ModuleType('tkinter.ttk')),
]:
    sys.modules.setdefault(_name, _mod)

sys.path.insert(0, 'rope')
import VideoManager as vm_module
from VideoManager import VideoManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EMB_DIM = 512
KPS_SHAPE = (5, 2)


def _unit_emb(idx):
    """Unit-norm embedding; each idx maps to a unique orthogonal direction."""
    emb = np.zeros(EMB_DIM, dtype=np.float32)
    emb[idx % EMB_DIM] = 1.0
    return emb


def _kps(value=0.0):
    return np.full(KPS_SHAPE, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def vm():
    return VideoManager()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def test_alpha_constant():
    assert hasattr(vm_module, 'KPS_SMOOTH_ALPHA')
    assert vm_module.KPS_SMOOTH_ALPHA == 0.6


def test_match_dist_constant():
    assert hasattr(vm_module, 'KPS_SMOOTH_MATCH_DIST')
    assert vm_module.KPS_SMOOTH_MATCH_DIST == 0.3


def test_max_faces_constant():
    assert hasattr(vm_module, 'KPS_SMOOTH_MAX_FACES')
    assert vm_module.KPS_SMOOTH_MAX_FACES == 10


# ---------------------------------------------------------------------------
# kps_smooth_state init
# ---------------------------------------------------------------------------
def test_kps_smooth_state_starts_empty(vm):
    assert hasattr(vm, 'kps_smooth_state')
    assert vm.kps_smooth_state == []


# ---------------------------------------------------------------------------
# _smooth_kps behaviour
# ---------------------------------------------------------------------------
def test_new_face_returns_raw_kps(vm):
    """First frame for a face: returns raw_kps unchanged and adds entry."""
    emb = _unit_emb(0)
    raw = _kps(5.0)
    result = vm._smooth_kps(emb, raw)

    np.testing.assert_array_equal(result, raw)
    assert len(vm.kps_smooth_state) == 1


def test_same_face_ewma_applied(vm):
    """Second frame with same face: result is EWMA blend."""
    emb = _unit_emb(0)
    raw1 = _kps(0.0)
    raw2 = _kps(10.0)

    vm._smooth_kps(emb, raw1)        # frame 1 — no smoothing
    result = vm._smooth_kps(emb, raw2)  # frame 2 — EWMA applied

    # expected = 0.6 * 10 + 0.4 * 0 = 6.0
    expected = 0.6 * raw2 + 0.4 * raw1
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_two_faces_tracked_independently(vm):
    """Different embeddings are stored as separate state entries."""
    emb_a = _unit_emb(0)
    emb_b = _unit_emb(1)   # orthogonal → cosine dist = 1 > 0.3 → new face

    vm._smooth_kps(emb_a, _kps(1.0))
    vm._smooth_kps(emb_b, _kps(2.0))

    assert len(vm.kps_smooth_state) == 2


def test_state_capped_at_max_faces(vm):
    """When full, adding a new face evicts the oldest entry."""
    max_f = vm_module.KPS_SMOOTH_MAX_FACES
    for i in range(max_f):
        vm._smooth_kps(_unit_emb(i), _kps(float(i)))

    assert len(vm.kps_smooth_state) == max_f

    vm._smooth_kps(_unit_emb(max_f), _kps(99.0))  # one more distinct face

    assert len(vm.kps_smooth_state) == max_f
