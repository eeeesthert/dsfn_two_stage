import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from abus_pairwise.models.encoder import ResNet50MultiScale


def test_encoder_supports_none_source():
    model = ResNet50MultiScale(pretrain_source="none")
    assert model is not None


def test_encoder_local_requires_ckpt():
    with pytest.raises(ValueError):
        ResNet50MultiScale(pretrain_source="local", checkpoint_path=None)
