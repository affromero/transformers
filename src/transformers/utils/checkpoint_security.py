"""Restricted checkpoint readers for the Torch 2.3 security backport."""

import codecs
import io
import pickle
import pickletools
from collections import OrderedDict

import numpy as np
from packaging.version import Version


def check_torch_load_security():
    import torch

    torch_version = Version(torch.__version__)
    patched_23 = (
        str(torch_version) == "2.3.0+cu121.security1"
        and hasattr(torch.serialization, "safe_globals")
        and hasattr(torch.serialization, "get_safe_globals")
    )
    if torch_version < Version("2.6.0") and not patched_23:
        raise ValueError("Checkpoint loading requires patched Torch 2.3.0+cu121.security1 or Torch >=2.6")


def restricted_torch_load(*args, **kwargs):
    import torch

    check_torch_load_security()
    if kwargs.get("weights_only", True) is not True:
        raise ValueError("Checkpoint loading requires weights_only=True")
    kwargs["weights_only"] = True
    return torch.load(*args, **kwargs)


def numpy_safe_globals():
    return [
        np.core.multiarray._reconstruct,
        np.core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]


class _NumpyCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            ("_codecs", "encode"): codecs.encode,
            ("builtins", "bytearray"): bytearray,
            ("__builtin__", "bytearray"): bytearray,
            ("collections", "OrderedDict"): OrderedDict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, identifier):
        raise pickle.UnpicklingError("Checkpoint persistent IDs are unsupported")


def load_numpy_checkpoint(handle):
    payload = handle.read()
    for opcode, argument, position in pickletools.genops(payload):
        if opcode.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
            raise pickle.UnpicklingError("Checkpoint extensions and persistent IDs are unsupported")
    return _NumpyCheckpointUnpickler(io.BytesIO(payload), encoding="latin1").load()


def load_rng_checkpoint(filename):
    import random

    import torch

    check_torch_load_security()
    with torch.serialization.safe_globals(numpy_safe_globals()):
        state = restricted_torch_load(filename, map_location="cpu")
    if not isinstance(state, dict) or not {"python", "numpy", "cpu"}.issubset(state):
        raise ValueError("Invalid checkpoint RNG state")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    torch.Generator().set_state(state["cpu"])
    return state
