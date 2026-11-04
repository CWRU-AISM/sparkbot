"""SAM3 loads the pinned checkpoint revision, not whatever facebook/sam3 main points at."""
import sys
import types

from spark_core import sam3 as core_sam3


def test_load_models_uses_pinned_checkpoint(monkeypatch):
    calls = {}

    def fake_download(repo, filename, revision=None):
        calls['download'] = (repo, filename, revision)
        return '/ckpt/sam3.pt'

    def fake_build(**kw):
        calls['build'] = kw
        return 'model'

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_confidence_threshold(self, t):
            self.t = t

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=fake_download))
    monkeypatch.setattr(core_sam3, 'build_sam3_image_model', fake_build)
    monkeypatch.setattr(core_sam3, 'Sam3Processor', FakeProcessor)
    monkeypatch.setattr(core_sam3, '_SAM3_IMPORT_ERROR', None)
    core_sam3.SPARKPerception().load_models(load_da3=False)
    assert calls['download'] == ('facebook/sam3', 'sam3.pt', core_sam3.SAM3_CKPT_REVISION)
    assert calls['build'] == {'checkpoint_path': '/ckpt/sam3.pt'}
    assert core_sam3.SAM3_CKPT_REVISION.startswith('3c879f39')
