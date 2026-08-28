"""Gemma 4 MLX-VLM adapter with strict text-only/chat-template policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import ArchitectureInfo, ForwardResult, LayerInfo
from .runtime import RuntimeUnavailableError, import_module, resolve_attribute


def _post_layer_token_index(cache: Any, next_offset: Any, sequence_length: int) -> int:
    """Return the first absolute position represented by a post-layer block.

    Ordinary Gemma layers retain their own cache offset; KV-sharing layers
    carry it as their returned shared ``next_offset``.  Both offsets have
    advanced through the block that just ran, so subtract its length.
    """

    offsets = [getattr(cache, "offset", 0), next_offset]
    cache_offset = max((int(value) for value in offsets if value is not None), default=0)
    return max(0, cache_offset - sequence_length)


class MlxGemma4Adapter:
    """Gemma-specific loading and architecture discovery; no silent checkpoint laxness."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.spec: Any | None = None
        self._architecture: ArchitectureInfo | None = None
        self._layers_path: str | None = None
        self._original_layers: list[Any] | None = None

    def load(self, spec: Any) -> None:
        if (
            not spec.text_only
            or spec.thinking
            or spec.speculative_decoding
            or spec.kv_cache_quantization
        ):
            raise ValueError(
                "Gemma v1 runs require text_only with thinking/speculation/KV quantization disabled"
            )
        module = import_module("mlx_vlm", "mlx-vlm")
        self.spec = spec
        source = str(spec.local_path or spec.repo_id)
        if spec.local_path is not None:
            # mlx-vlm treats a supplied revision as a Hub reference even when
            # ``source`` is an immutable local snapshot.  Loading that path
            # directly keeps the already-recorded revision as provenance and
            # avoids a second network/repository resolution.
            self.model, self.processor = module.load(source)
        else:
            try:
                self.model, self.processor = module.load(source, revision=spec.revision)
            except TypeError:
                self.model, self.processor = module.load(source)
        self._architecture, self._layers_path = self._discover_architecture()

    def unload(self) -> None:
        self.unwrap_layers()
        self.model = None
        self.processor = None
        self.spec = None
        self._architecture = None
        self._layers_path = None

    def format_prompt(self, messages: Sequence[dict[str, str]]) -> list[int]:
        if self.model is None or self.processor is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        if (
            len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise ValueError("Gemma v1 requires explicit system and user chat messages")
        prompt_utils = import_module("mlx_vlm.prompt_utils", "mlx-vlm")
        # The public helper is required for instruction checkpoints; raw strings are forbidden.
        rendered = prompt_utils.apply_chat_template(
            self.processor,
            self.model.config,
            list(messages),
            chat_template_kwargs={"enable_thinking": False},
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return list(tokenizer.encode(rendered))

    def architecture(self) -> ArchitectureInfo:
        if self._architecture is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        return self._architecture

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.processor is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return str(tokenizer.decode(list(token_ids)))

    def eos_token_ids(self) -> set[int]:
        if self.processor is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        result: set[int]
        ids = getattr(tokenizer, "eos_token_ids", None)
        if ids is not None:
            if isinstance(ids, int):
                result = {ids}
            else:
                result = {int(value) for value in ids}
        else:
            eos = getattr(tokenizer, "eos_token_id", None)
            result = {int(eos)} if eos is not None else set()
        # Gemma instruction templates terminate assistant turns with this
        # distinct special token rather than the tokenizer's normal EOS.
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                terminator = list(encode("<turn|>", add_special_tokens=False))
            except (TypeError, ValueError):
                terminator = []
            if len(terminator) == 1:
                result.add(int(terminator[0]))
        return result

    def wrap_layers(self, controller: Any) -> None:
        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        if self._original_layers is not None:
            raise RuntimeError("Layers are already wrapped")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        mlx_nn = import_module("mlx.nn", "mlx")

        class InstrumentedLayer(mlx_nn.Module):
            def __init__(self, base: Any, index: int) -> None:
                super().__init__()
                self.base = base
                self.index = index

            def __getattr__(self, name: str) -> Any:
                if name in {"base", "index"}:
                    return super().__getattr__(name)
                return getattr(self.base, name)

            def __call__(
                self,
                x: Any,
                mask: Any = None,
                cache: Any = None,
                per_layer_input: Any = None,
                shared_kv: Any = None,
                offset: Any = None,
            ) -> tuple[Any, Any, Any]:
                hidden, next_shared_kv, next_offset = self.base(
                    x,
                    mask,
                    cache,
                    per_layer_input=per_layer_input,
                    shared_kv=shared_kv,
                    offset=offset,
                )
                # The cache offset advances inside attention.  Most Gemma
                # layers expose it on their own cache, whereas KV-sharing
                # layers receive and return it through ``offset``.  Looking
                # only at the per-layer cache silently made interventions in
                # the latter group inactive because their cache is ``None``.
                token_index = _post_layer_token_index(
                    cache, next_offset, int(hidden.shape[-2])
                )
                hidden = controller.apply_post_layer_mlx(self.index, hidden, token_index)
                return hidden, next_shared_kv, next_offset

        self._original_layers = layers
        setattr(
            parent,
            attribute,
            [InstrumentedLayer(layer, index) for index, layer in enumerate(layers)],
        )

    def unwrap_layers(self) -> None:
        if self._original_layers is None or self._layers_path is None or self.model is None:
            return
        parent, attribute = self._layer_parent_and_attribute()
        setattr(parent, attribute, self._original_layers)
        self._original_layers = None

    def make_cache(self) -> Any:
        if self.model is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        return self.model.make_cache()

    def forward(self, token_ids: Sequence[int], cache: Any | None = None) -> ForwardResult:
        if self.model is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        mx = import_module("mlx.core", "mlx")
        output = self.model(mx.array([list(token_ids)]), cache=cache)
        logits = output.logits if hasattr(output, "logits") else output
        if cache is None:
            mx.eval(logits)
        else:
            mx.eval(logits, [entry.state for entry in cache if hasattr(entry, "state")])
        return ForwardResult(
            logits=np.asarray(logits.astype(mx.float32)), hidden_summaries={}, cache=cache
        )

    def _discover_architecture(self) -> tuple[ArchitectureInfo, str]:
        if self.model is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        path, layers_value = resolve_attribute(self.model, ("language_model.model.layers",))
        layers = list(layers_value)
        if not layers:
            raise RuntimeUnavailableError("Gemma decoder layer collection is empty")
        config = getattr(self.model.config, "text_config", None)
        hidden_width = int(getattr(config, "hidden_size", 0))
        vocabulary_size = int(getattr(config, "vocab_size", 0))
        if hidden_width <= 0 or vocabulary_size <= 0:
            raise RuntimeUnavailableError("Gemma configuration lacks hidden_size or vocab_size")
        count = len(layers)
        layer_info = tuple(
            LayerInfo(
                index=index,
                normalized_depth=index / max(1, count - 1),
                layer_type=getattr(layer, "layer_type", type(layer).__name__),
                attention_kind=getattr(layer, "attention_type", "unknown"),
                kv_sharing_role=getattr(layer, "kv_sharing_role", None),
            )
            for index, layer in enumerate(layers)
        )
        return (
            ArchitectureInfo(
                decoder_layer_path=path,
                hidden_width=hidden_width,
                vocabulary_size=vocabulary_size,
                layers=layer_info,
                final_norm_path="language_model.model.norm",
                output_head_path="language_model.logits_from_hidden",
                cache_type="mlx_vlm Gemma4 LanguageModel.make_cache",
                cache_count=count,
                has_per_layer_inputs=True,
            ),
            path,
        )

    def _layer_parent_and_attribute(self) -> tuple[Any, str]:
        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        parts = self._layers_path.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
