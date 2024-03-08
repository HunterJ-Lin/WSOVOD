# ensure the builtin datasets are registered
from . import datasets  # isort:skip

from .build import build_detection_test_loader, build_detection_train_loader

__all__ = [k for k in globals().keys() if not k.startswith("_")]
