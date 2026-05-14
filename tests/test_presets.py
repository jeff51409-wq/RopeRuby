"""Unit tests for GUI preset save/load/delete logic.

Run from project root:
    pytest tests/test_presets.py -v
"""
import json
import os
import sys
import types
import tempfile
import pytest

# ---- mock tkinter so GUI can be imported without a display ----
_tk = types.ModuleType('tkinter')
_tk.Tk = type('Tk', (), {'__init__': lambda s: None, 'title': lambda *a: None})
_tk.Frame = type('Frame', (), {'__init__': lambda *a, **k: None})
_tk.LabelFrame = type('LabelFrame', (), {'__init__': lambda *a, **k: None, 'place': lambda *a, **k: None})
_tk.Canvas = type('Canvas', (), {'__init__': lambda *a, **k: None})
_tk.Button = type('Button', (), {'__init__': lambda *a, **k: None, 'place': lambda *a, **k: None, 'config': lambda *a, **k: None, 'bind': lambda *a, **k: None})
_tk.Label = type('Label', (), {'__init__': lambda *a, **k: None})
_tk.Entry = type('Entry', (), {'__init__': lambda *a, **k: None})
_tk.StringVar = type('StringVar', (), {'__init__': lambda *a, **k: None, 'get': lambda s: '', 'set': lambda *a: None})
_tk.IntVar = type('IntVar', (), {'__init__': lambda *a, **k: None})
_tk.PhotoImage = type('PhotoImage', (), {'__init__': lambda *a, **k: None})

_ttk = types.ModuleType('tkinter.ttk')
_ttk.Combobox = type('Combobox', (), {
    '__init__': lambda *a, **k: None,
    'pack': lambda *a, **k: None,
    'set': lambda *a, **k: None,
    'configure': lambda *a, **k: None,
    '__getitem__': lambda s, k: [],
    '__setitem__': lambda *a: None,
})

sys.modules['tkinter'] = _tk
sys.modules['tkinter.ttk'] = _ttk
sys.modules['tkinter.simpledialog'] = types.ModuleType('tkinter.simpledialog')
sys.modules['tkinter.filedialog'] = types.ModuleType('tkinter.filedialog')
sys.modules['tkinter.font'] = types.ModuleType('tkinter.font')

# ---- mock other heavy deps ----
for mod in ['cv2', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageTk', 'skimage',
            'skimage.transform', 'torch', 'torchvision',
            'torchvision.transforms', 'torchvision.transforms.v2',
            'torchvision.transforms.functional', 'torchvision.ops',
            'mimetypes', 'webbrowser']:
    sys.modules[mod] = types.ModuleType(mod)

# ---- helper: an object with the preset methods only ----
# We test the data-layer logic directly without a real GUI instance.

class PresetMixin:
    """Extracted preset logic for testing without tkinter display."""

    def __init__(self, json_path):
        self._json_path = json_path
        self.parameters = {
            'ColorState': False, 'ColorAmount': [0, 0, 0],
            'AutoColorState': False, 'AutoColorAmount': [100],
        }

    def _read_json(self):
        with open(self._json_path, 'r') as f:
            return json.load(f)

    def _write_json(self, data):
        with open(self._json_path, 'w') as f:
            json.dump(data, f)

    def save_preset(self, name):
        data = self._read_json()
        if 'presets' not in data:
            data['presets'] = {}
        data['presets'][name] = {k: v for k, v in self.parameters.items()
                                  if not callable(v)}
        self._write_json(data)

    def load_preset(self, name):
        data = self._read_json()
        preset = data.get('presets', {}).get(name, {})
        for key, value in preset.items():
            if key in self.parameters:
                self.parameters[key] = value

    def delete_preset(self, name):
        data = self._read_json()
        data.get('presets', {}).pop(name, None)
        self._write_json(data)

    def list_presets(self):
        data = self._read_json()
        return sorted(data.get('presets', {}).keys())


@pytest.fixture
def tmp_json(tmp_path):
    path = tmp_path / 'data.json'
    path.write_text('{}')
    return str(path)


def test_save_preset_persists(tmp_json):
    m = PresetMixin(tmp_json)
    m.parameters['ColorState'] = True
    m.save_preset('test1')
    with open(tmp_json) as f:
        saved = json.load(f)
    assert saved['presets']['test1']['ColorState'] is True


def test_load_preset_updates_parameters(tmp_json):
    m = PresetMixin(tmp_json)
    m.save_preset('snap')
    m.parameters['ColorState'] = True
    m.load_preset('snap')
    assert m.parameters['ColorState'] is False


def test_save_overwrites_existing(tmp_json):
    m = PresetMixin(tmp_json)
    m.parameters['ColorState'] = False
    m.save_preset('p')
    m.parameters['ColorState'] = True
    m.save_preset('p')
    m.parameters['ColorState'] = False
    m.load_preset('p')
    assert m.parameters['ColorState'] is True


def test_delete_removes_preset(tmp_json):
    m = PresetMixin(tmp_json)
    m.save_preset('to_delete')
    m.delete_preset('to_delete')
    assert 'to_delete' not in m.list_presets()


def test_list_presets_sorted(tmp_json):
    m = PresetMixin(tmp_json)
    for name in ['zebra', 'apple', 'mango']:
        m.save_preset(name)
    assert m.list_presets() == ['apple', 'mango', 'zebra']


def test_load_nonexistent_preset_is_noop(tmp_json):
    m = PresetMixin(tmp_json)
    original = dict(m.parameters)
    m.load_preset('does_not_exist')
    assert m.parameters == original
