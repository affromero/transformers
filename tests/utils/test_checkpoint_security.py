import io
import os
import pickle
import random

import numpy as np
import pytest
import torch

from transformers import BertConfig, BertForSequenceClassification, BertModel, Trainer, TrainingArguments
from transformers.utils.checkpoint_security import (
    check_torch_load_security,
    load_numpy_checkpoint,
    restricted_torch_load,
)


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_numpy_weight_checkpoints_preserve_arrays_and_nested_metadata(protocol):
    weights = np.arange(12, dtype=np.float32).reshape(3, 4)
    checkpoint = {"model": {"linear.weight": weights}, "weights": [weights, (np.float32(3.5),)], "__author__": "test"}
    loaded = load_numpy_checkpoint(io.BytesIO(pickle.dumps(checkpoint, protocol=protocol)))
    np.testing.assert_array_equal(loaded["model"]["linear.weight"], weights)
    np.testing.assert_array_equal(loaded["weights"][0], weights)
    assert loaded["weights"][1][0] == np.float32(3.5)
    assert loaded["__author__"] == "test"


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_numpy_checkpoints_reject_executable_objects_before_side_effects(tmp_path, protocol):
    marker = tmp_path / "executed"

    class Executable:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    payload = pickle.dumps({"weights": Executable()}, protocol=protocol)
    with pytest.raises(pickle.UnpicklingError, match="Unsupported checkpoint global"):
        load_numpy_checkpoint(io.BytesIO(payload))
    assert not marker.exists()


@pytest.mark.parametrize("framework", ["tensorflow", "flax"])
@pytest.mark.parametrize("sharded", [False, True])
def test_cross_framework_binary_checkpoints_preserve_inference_and_reject_unpatched_torch(
    tmp_path, monkeypatch, framework, sharded
):
    config = BertConfig(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    original = BertModel(config).eval()
    original.save_pretrained(tmp_path, safe_serialization=False, max_shard_size="1KB" if sharded else "10MB")
    inputs = np.array([[1, 2, 3]], dtype=np.int32)
    with torch.no_grad():
        expected = original(torch.tensor(inputs.astype(np.int64))).last_hidden_state.numpy()
    if framework == "tensorflow":
        from transformers import TFBertModel

        model_class = TFBertModel
    else:
        from transformers import FlaxBertModel

        model_class = FlaxBertModel
    loaded = model_class.from_pretrained(tmp_path, from_pt=True)
    actual = np.asarray(loaded(inputs).last_hidden_state)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    monkeypatch.setattr(torch, "__version__", "2.3.0")
    with pytest.raises(ValueError, match="patched Torch"):
        model_class.from_pretrained(tmp_path, from_pt=True)


def test_training_resumes_optimizer_scheduler_and_model_state(tmp_path):
    config = BertConfig(
        vocab_size=12,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    dataset = [
        {"input_ids": torch.tensor([1, 2 + index % 3, 3]), "labels": torch.tensor(index % 2)} for index in range(8)
    ]

    def training_arguments(directory):
        return TrainingArguments(
            output_dir=str(directory),
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
            max_steps=2,
            save_steps=1,
            per_device_train_batch_size=2,
            learning_rate=0.001,
            dataloader_pin_memory=False,
        )

    complete = Trainer(
        model=BertForSequenceClassification(config),
        args=training_arguments(tmp_path / "complete"),
        train_dataset=dataset,
    )
    complete.train()
    resumed = Trainer(
        model=BertForSequenceClassification(config),
        args=training_arguments(tmp_path / "resumed"),
        train_dataset=dataset,
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "complete" / "checkpoint-1"))
    assert resumed.state.global_step == 2
    assert resumed.lr_scheduler.state_dict() == complete.lr_scheduler.state_dict()
    for name, parameter in complete.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], parameter)
    for state in resumed.optimizer.state_dict()["state"].values():
        assert state["step"] == 2


def test_numpy_checkpoints_reject_cached_extension_globals():
    with pytest.raises(pickle.UnpicklingError, match="extensions"):
        load_numpy_checkpoint(io.BytesIO(b"\x80\x02\x82\x01."))


def test_torch_loading_rejects_unpatched_23_even_with_newer_apis(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(2)}, checkpoint)
    monkeypatch.setattr(torch, "__version__", "2.3.0")
    with pytest.raises(ValueError, match="patched Torch"):
        restricted_torch_load(checkpoint)


def test_torch_loading_rejects_unsafe_override(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(2)}, checkpoint)
    with pytest.raises(ValueError, match="weights_only"):
        restricted_torch_load(checkpoint, weights_only=False)


@pytest.mark.parametrize(
    "version",
    ["2.6.0.dev20241001", "2.6.0a0", "2.7.0a0", "2.8.0.dev0", "2.3.0+cu121", "2.3.0+git1234567", "2.3.0+security.0"],
)
def test_checkpoint_loading_rejects_prereleases_and_unpatched_builds(monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(ValueError, match="patched Torch"):
        check_torch_load_security()


@pytest.mark.parametrize("version", ["2.3.0+cu121.security1", "2.3.0+security.1"])
@pytest.mark.parametrize("api", ["safe_globals", "get_safe_globals"])
def test_checkpoint_loading_requires_restricted_apis_in_the_23_backport(monkeypatch, version, api):
    monkeypatch.setattr(torch, "__version__", version)
    check_torch_load_security()
    monkeypatch.delattr(torch.serialization, api)
    with pytest.raises(ValueError, match="patched Torch"):
        check_torch_load_security()


@pytest.mark.parametrize("version", ["2.3.0+cu121.security1", "2.3.0+security.1"])
def test_maintained_builds_read_tensor_checkpoints_without_unrestricted_loading(tmp_path, monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    checkpoint = tmp_path / "weights.pt"
    expected = {"weight": torch.arange(6).reshape(2, 3), "label": "maintained"}
    torch.save(expected, checkpoint)
    actual = restricted_torch_load(checkpoint, map_location="cpu")
    torch.testing.assert_close(actual["weight"], expected["weight"])
    assert actual["label"] == expected["label"]


def test_binary_model_checkpoints_preserve_inference(tmp_path):
    model = BertModel(
        BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).eval()
    model.save_pretrained(tmp_path, safe_serialization=False)
    loaded = BertModel.from_pretrained(tmp_path).eval()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(inputs).last_hidden_state, model(inputs).last_hidden_state)


def test_trainer_rng_checkpoints_restore_all_cpu_generators_without_global_allowlists(tmp_path):
    trainer = Trainer(
        model=torch.nn.Linear(2, 1),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    before = torch.serialization.get_safe_globals()
    trainer._save_rng_state(str(tmp_path))
    expected = random.random(), np.random.random(), torch.rand(3)
    trainer._load_rng_state(str(tmp_path))
    actual = random.random(), np.random.random(), torch.rand(3)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    assert set(torch.serialization.get_safe_globals()) == set(before)


def test_trainer_rng_checkpoints_cannot_execute_pickle_payloads(tmp_path):
    marker = tmp_path / "rng-executed"

    class Executable:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    torch.save({"python": Executable()}, tmp_path / "rng_state.pth")
    trainer = Trainer(
        model=torch.nn.Linear(2, 1),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    with pytest.raises(pickle.UnpicklingError):
        trainer._load_rng_state(str(tmp_path))
    assert not marker.exists()
