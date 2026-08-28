from pathlib import Path
from types import SimpleNamespace

import pytest

from lspe.models.runtime import RuntimeUnavailableError, import_module


def test_gemma_post_layer_position_uses_shared_kv_offset() -> None:
    from lspe.models.mlx_gemma4 import _post_layer_token_index

    assert _post_layer_token_index(SimpleNamespace(offset=17), None, 1) == 16
    # KV-sharing layers have no independent cache, but their returned offset
    # still identifies the generated token's absolute position.
    assert _post_layer_token_index(None, 17, 1) == 16


def test_missing_runtime_has_actionable_diagnostic() -> None:
    with pytest.raises(RuntimeUnavailableError, match="preflight"):
        import_module("lspe_missing_runtime_for_test", "mlx-lm")


def test_gemma_adapter_accepts_scalar_eos_token_ids() -> None:
    from lspe.models.mlx_gemma4 import MlxGemma4Adapter

    adapter = MlxGemma4Adapter()
    adapter.processor = type(
        "Processor", (), {"tokenizer": type("Tokenizer", (), {"eos_token_ids": 42})()}
    )()
    assert adapter.eos_token_ids() == {42}


def test_gemma_adapter_stops_on_chat_turn_terminator() -> None:
    from lspe.models.mlx_gemma4 import MlxGemma4Adapter

    tokenizer = type(
        "Tokenizer",
        (),
        {"eos_token_id": 1, "encode": staticmethod(lambda _text, **_kwargs: [106])},
    )()
    adapter = MlxGemma4Adapter()
    adapter.processor = type("Processor", (), {"tokenizer": tokenizer})()
    assert adapter.eos_token_ids() == {1, 106}


@pytest.mark.parametrize(
    ("module_name", "adapter_name", "runtime_module", "attributes"),
    [
        ("lspe.models.mlx_gemma4", "MlxGemma4Adapter", "mlx_vlm", ("model", "processor")),
        ("lspe.models.mlx_qwen3", "MlxQwen3Adapter", "mlx_lm", ("model", "tokenizer")),
    ],
)
def test_mlx_adapters_load_local_snapshot_without_revision(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    adapter_name: str,
    runtime_module: str,
    attributes: tuple[str, str],
) -> None:
    module = __import__(module_name, fromlist=[adapter_name])
    adapter = getattr(module, adapter_name)()
    calls: list[tuple[str, dict[str, object]]] = []

    def load(source: str, **kwargs: object) -> tuple[object, object]:
        calls.append((source, kwargs))
        return object(), object()

    monkeypatch.setattr(module, "import_module", lambda name, _package: SimpleNamespace(load=load))
    monkeypatch.setattr(adapter, "_discover_architecture", lambda: (object(), "layers"))
    spec = SimpleNamespace(
        local_path=Path("/tmp/immutable-model-snapshot"),
        repo_id="owner/model",
        revision="immutable-commit",
        text_only=True,
        thinking=False,
        speculative_decoding=False,
        kv_cache_quantization=False,
    )

    adapter.load(spec)

    assert calls == [(str(spec.local_path), {})]
    assert getattr(adapter, attributes[0]) is not None
    assert getattr(adapter, attributes[1]) is not None
