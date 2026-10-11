import pytest

from transformers.configuration_utils import get_configuration_file
from transformers.dynamic_module_utils import get_imports
from transformers.models.clvp.number_normalizer import EnglishNormalizer
from transformers.models.deprecated.gptsan_japanese.tokenization_gptsan_japanese import (
    SubWordJapaneseTokenizer as GPTSanTokenizer,
)
from transformers.models.gpt_neox_japanese.tokenization_gpt_neox_japanese import (
    SubWordJapaneseTokenizer as NeoXTokenizer,
)
from transformers.models.marian.tokenization_marian import MarianTokenizer
from transformers.models.nougat.tokenization_nougat_fast import normalize_list_like_lines


def test_import_parser_handles_attribute_conditions_and_ignores_optional_imports(tmp_path):
    module = tmp_path / "model.py"
    module.write_text(
        "import os\nif torch.cuda.is_available():\n    import torch.nn\n"
        "try:\n    import optional_backend\nexcept ImportError:\n    pass\n"
        "if is_flash_attn_2_available():\n    from flash_attn import flash_attn_func\n"
    )
    assert get_imports(module) == ["os", "torch"]


@pytest.mark.timeout(3)
def test_import_parser_handles_long_unterminated_try_text(tmp_path):
    module = tmp_path / "model.py"
    module.write_text("import os\n" + "# try : " * 10000)
    assert get_imports(module) == ["os"]


@pytest.mark.timeout(3)
def test_marian_language_prefix_preserves_plain_text_and_rejects_unterminated_codes():
    assert MarianTokenizer.remove_language_code(None, ">>fr<<Bonjour") == ([">>fr<<"], "Bonjour")
    text = ">>" * 10000
    assert MarianTokenizer.remove_language_code(None, text) == ([], text)


@pytest.mark.timeout(3)
@pytest.mark.parametrize("tokenizer_class", [GPTSanTokenizer, NeoXTokenizer])
def test_japanese_price_normalization_handles_adversarial_numbers(tokenizer_class):
    tokenizer = tokenizer_class({"x": 0}, {0: "x"}, {})
    assert tokenizer.clean_text("100円") == "<PRICE>"
    assert tokenizer.clean_text("1,234,567円") == "<PRICE>"
    text = "1" * 100000 + "!"
    try:
        normalized = tokenizer.clean_text(text)
    except TimeoutError as error:
        assert "timed out" in str(error)
    else:
        assert normalized.endswith("!")


@pytest.mark.timeout(3)
def test_number_normalization_preserves_prices_and_handles_long_non_numbers():
    normalizer = EnglishNormalizer()
    assert "dollars" in normalizer("$12")
    text = "," * 10000
    assert normalizer(text) == text


@pytest.mark.timeout(3)
def test_number_normalization_bounds_unanchored_decimal_search():
    with pytest.raises(TimeoutError):
        EnglishNormalizer()("1" * 100000 + "!")


@pytest.mark.timeout(3)
def test_nougat_list_normalization_handles_long_plain_lines():
    text = "i" * 10000 + "!"
    assert normalize_list_like_lines(text) == text


@pytest.mark.timeout(3)
def test_configuration_selection_ignores_long_non_config_filenames():
    assert get_configuration_file(["x" * 10000, "config.4.40.0.json", "config.99.0.0.json"]) == "config.4.40.0.json"


@pytest.mark.timeout(3)
def test_weight_decay_preserves_regex_matching_and_bounds_expensive_patterns():
    from transformers.optimization_tf import AdamWeightDecay

    optimizer = AdamWeightDecay(weight_decay_rate=0.1, include_in_weight_decay=[r"kernel$"])
    assert optimizer._do_use_weight_decay("layer/kernel")
    assert optimizer._do_use_weight_decay("layer/bias")
    optimizer = AdamWeightDecay(weight_decay_rate=0.1, exclude_from_weight_decay=[r"bias$"])
    assert not optimizer._do_use_weight_decay("layer/bias")
    optimizer = AdamWeightDecay(weight_decay_rate=0.1, include_in_weight_decay=[r"(a+)+$"])
    with pytest.raises(TimeoutError):
        optimizer._do_use_weight_decay("a" * 100000 + "!")


@pytest.mark.timeout(3)
def test_donut_tokens_preserve_structured_data_and_bound_unterminated_tags():
    from transformers.models.donut.processing_donut import DonutProcessor

    class Parser:
        token2json = DonutProcessor.token2json

    parser = Parser()
    assert parser.token2json("<s_name>Alice</s_name><s_age>20</s_age>", added_vocab={}) == {
        "name": "Alice",
        "age": "20",
    }
    text = "<s_" * 20000
    assert parser.token2json(text, added_vocab={}) == {"text_sequence": text}


def test_mobilevit_yaml_preserves_nested_configuration_and_rejects_executable_tags(tmp_path):
    import yaml

    from transformers.models.mobilevitv2.convert_mlcvnets_to_pytorch import load_orig_config_file

    path = tmp_path / "model.yaml"
    path.write_text("model:\n  classification:\n    n_classes: 3\n")
    assert getattr(load_orig_config_file(path), "model.classification.n_classes") == 3
    marker = tmp_path / "executed"
    path.write_text("!!python/object/apply:os.system ['touch " + str(marker) + "']")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_orig_config_file(path)
    assert not marker.exists()
    path.write_text("- invalid\n- mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_orig_config_file(path)


@pytest.mark.timeout(3)
def test_tf_weight_names_preserve_conversion_and_reject_pathological_separators():
    from transformers.modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name

    name, transpose = convert_tf_weight_name_to_pt_weight_name("model/layer/kernel:0", tf_weight_shape=(2, 3))
    assert name == "layer.weight"
    for name in ["a" * 10000 + "___b", "a___" * 10000]:
        with pytest.raises(ValueError, match="too long"):
            convert_tf_weight_name_to_pt_weight_name(name, tf_weight_shape=(2, 3))


@pytest.mark.timeout(3)
def test_doctest_preprocessing_preserves_examples_and_bounds_newline_floods():
    from transformers.testing_utils import preprocess_string

    text = "```python\n>>> from datasets import load_dataset\n>>> value = load_dataset('example')\n```"
    result = preprocess_string(text, skip_cuda_tests=False)
    assert "value = load_dataset('example')" in result
    assert "# doctest: +IGNORE_RESULT" in result
    with pytest.raises(TimeoutError):
        preprocess_string("```python\n" + "\n" * 100000 + "!", skip_cuda_tests=False)
