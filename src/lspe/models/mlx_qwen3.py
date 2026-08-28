"""MLX-LM Qwen text adapter; version-sensitive details stay confined here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import ArchitectureInfo, ForwardResult, LayerInfo
from .runtime import RuntimeUnavailableError, import_module, resolve_attribute


class MlxQwen3Adapter:
    """Adapter shell for Qwen, used as the reference/fallback text backend."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._architecture: ArchitectureInfo | None = None
        self._layers_path: str | None = None
        self._original_layers: list[Any] | None = None
        self._original_output_projections: dict[int, Any] = {}
        self._original_attentions: dict[int, Any] = {}
        self._original_transform_attentions: dict[int, Any] = {}

    def load(self, spec: Any) -> None:
        mlx_lm = import_module("mlx_lm", "mlx-lm")
        source = str(spec.local_path or spec.repo_id)
        if spec.local_path is not None:
            # A resolved local snapshot is already immutable; passing a
            # revision makes some mlx-lm profiles attempt Hub resolution.
            self.model, self.tokenizer = mlx_lm.load(source)
        else:
            try:
                self.model, self.tokenizer = mlx_lm.load(source, revision=spec.revision)
            except TypeError:
                # Older reviewed profiles did not expose `revision`; this is recorded by preflight.
                self.model, self.tokenizer = mlx_lm.load(source)
        self._architecture, self._layers_path = self._discover_architecture()

    def unload(self) -> None:
        self.unwrap_attention_transformer()
        self.unwrap_attention_observer()
        self.unwrap_layers()
        self.model = None
        self.tokenizer = None
        self._architecture = None
        self._layers_path = None

    def format_prompt(self, messages: Sequence[dict[str, str]]) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        rendered = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        return list(self.tokenizer.encode(rendered))

    def architecture(self) -> ArchitectureInfo:
        if self._architecture is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        return self._architecture

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        return str(self.tokenizer.decode(list(token_ids)))

    def eos_token_ids(self) -> set[int]:
        if self.tokenizer is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        ids = getattr(self.tokenizer, "eos_token_ids", None)
        if ids is not None:
            if isinstance(ids, int):
                return {ids}
            return {int(value) for value in ids}
        eos = getattr(self.tokenizer, "eos_token_id", None)
        return {int(eos)} if eos is not None else set()

    def wrap_layers(self, controller: Any) -> None:
        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        if self._original_layers is not None:
            raise RuntimeError("Layers are already wrapped")
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

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                hidden = self.base(*args, **kwargs)
                cache = kwargs.get("cache")
                if cache is None and len(args) >= 3:
                    cache = args[2]
                cache_offset = int(getattr(cache, "offset", 0))
                token_index = max(0, cache_offset - hidden.shape[-2])
                return controller.apply_post_layer_mlx(self.index, hidden, token_index)

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

    def wrap_attention_observer(
        self, observer: Any, selected_layers: frozenset[int]
    ) -> None:
        """Observe per-head residual contributions without changing model output."""

        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        if self._original_output_projections:
            raise RuntimeError("Attention output projections are already observed")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        invalid = sorted(selected_layers - set(range(len(layers))))
        if invalid:
            raise ValueError(f"Attention observer selected invalid layers: {invalid}")
        mlx_nn = import_module("mlx.nn", "mlx")
        mx = import_module("mlx.core", "mlx")

        class ObservedOutputProjection(mlx_nn.Module):
            def __init__(self, base: Any, layer_index: int, head_count: int) -> None:
                super().__init__()
                self.base = base
                self.layer_index = layer_index
                self.head_count = head_count
                self.weight = _dequantized_linear_weight(mx, base)

            def __call__(self, x: Any) -> Any:
                output = self.base(x)
                batch, length, width = x.shape
                if width % self.head_count:
                    raise RuntimeError("Attention projection width is not divisible by heads")
                head_width = width // self.head_count
                heads = x.reshape(batch, length, self.head_count, head_width).astype(
                    mx.float32
                )
                weights = self.weight.reshape(
                    self.weight.shape[0], self.head_count, head_width
                ).astype(mx.float32)
                contributions = mx.einsum("blhd,ohd->blho", heads, weights)
                mx.eval(contributions)
                observer.record_mlx(self.layer_index, contributions)
                return output

        class ObservedAttention(mlx_nn.Module):
            def __init__(self, base: Any, layer_index: int) -> None:
                super().__init__()
                self.base = base
                self.layer_index = layer_index

            def __getattr__(self, name: str) -> Any:
                if name in {"base", "layer_index"}:
                    return super().__getattr__(name)
                return getattr(self.base, name)

            def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
                output = self.base(x, mask=mask, cache=cache)
                if cache is None and hasattr(observer, "record_attention_mlx"):
                    batch, length, _ = x.shape
                    queries = self.base.q_proj(x)
                    keys = self.base.k_proj(x)
                    queries = self.base.q_norm(
                        queries.reshape(batch, length, self.base.n_heads, -1)
                    ).transpose(0, 2, 1, 3)
                    keys = self.base.k_norm(
                        keys.reshape(batch, length, self.base.n_kv_heads, -1)
                    ).transpose(0, 2, 1, 3)
                    queries = self.base.rope(queries)
                    keys = self.base.rope(keys)
                    repeats = self.base.n_heads // self.base.n_kv_heads
                    keys = mx.repeat(keys, repeats, axis=1)
                    scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.base.scale
                    if isinstance(mask, str) and mask == "causal":
                        causal = mx.triu(
                            mx.full((length, length), float("-inf"), dtype=mx.float32),
                            k=1,
                        )
                        scores = scores + causal
                    elif mask is not None:
                        scores = scores + mask
                    patterns = mx.softmax(scores.astype(mx.float32), axis=-1)
                    mx.eval(patterns)
                    observer.record_attention_mlx(self.layer_index, patterns)
                return output

        for layer_index in sorted(selected_layers):
            attention = layers[layer_index].self_attn
            projection = attention.o_proj
            self._original_output_projections[layer_index] = projection
            attention.o_proj = ObservedOutputProjection(
                projection, layer_index, int(attention.n_heads)
            )
            self._original_attentions[layer_index] = attention
            layers[layer_index].self_attn = ObservedAttention(attention, layer_index)

    def unwrap_attention_observer(self) -> None:
        if not self._original_output_projections and not self._original_attentions:
            return
        if self.model is None or self._layers_path is None:
            raise RuntimeError("Cannot restore attention projections after model unload")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        for layer_index, attention in self._original_attentions.items():
            layers[layer_index].self_attn = attention
        self._original_attentions.clear()
        for layer_index, projection in self._original_output_projections.items():
            layers[layer_index].self_attn.o_proj = projection
        self._original_output_projections.clear()

    def wrap_attention_transformer(
        self, transformer: Any, selected_layers: frozenset[int]
    ) -> None:
        """Replace cached decode attention with an inspectable score transformation."""

        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        if self._original_transform_attentions or self._original_attentions:
            raise RuntimeError("Attention modules are already wrapped")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        invalid = sorted(selected_layers - set(range(len(layers))))
        if invalid:
            raise ValueError(f"Attention transformer selected invalid layers: {invalid}")
        mlx_nn = import_module("mlx.nn", "mlx")
        mx = import_module("mlx.core", "mlx")

        class TransformedAttention(mlx_nn.Module):
            def __init__(self, base: Any, layer_index: int) -> None:
                super().__init__()
                self.base = base
                self.layer_index = layer_index

            def __getattr__(self, name: str) -> Any:
                if name in {"base", "layer_index"}:
                    return super().__getattr__(name)
                return getattr(self.base, name)

            def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
                # Decode-only transformers must not replace multi-token prefill:
                # a zero attention bias would otherwise discard its causal mask.
                if cache is None or x.shape[1] != 1:
                    return self.base(x, mask=mask, cache=cache)
                batch, length, _ = x.shape
                queries = self.base.q_proj(x)
                keys = self.base.k_proj(x)
                values = self.base.v_proj(x)
                queries = self.base.q_norm(
                    queries.reshape(batch, length, self.base.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                keys = self.base.k_norm(
                    keys.reshape(batch, length, self.base.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                values = values.reshape(
                    batch, length, self.base.n_kv_heads, -1
                ).transpose(0, 2, 1, 3)
                queries = self.base.rope(queries, offset=cache.offset)
                keys = self.base.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
                repeats = self.base.n_heads // self.base.n_kv_heads
                expanded_keys = mx.repeat(keys, repeats, axis=1)
                scores = (
                    queries @ expanded_keys.transpose(0, 1, 3, 2)
                ) * self.base.scale
                if isinstance(mask, str) and mask != "causal":
                    raise RuntimeError(f"Unsupported cached attention mask: {mask}")
                transformed = transformer.apply_mlx(self.layer_index, scores)
                attention_bias = transformed - scores
                if mask is not None and not isinstance(mask, str):
                    attention_bias = attention_bias + mask
                has_effect = getattr(transformer, "has_effect", lambda _: True)(
                    self.layer_index
                )
                output = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=self.base.scale,
                    mask=attention_bias if has_effect else mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
                return self.base.o_proj(output)

        for layer_index in sorted(selected_layers):
            attention = layers[layer_index].self_attn
            self._original_transform_attentions[layer_index] = attention
            layers[layer_index].self_attn = TransformedAttention(attention, layer_index)

    def unwrap_attention_transformer(self) -> None:
        """Restore attention modules replaced by `wrap_attention_transformer`."""

        if not self._original_transform_attentions:
            return
        if self.model is None or self._layers_path is None:
            raise RuntimeError("Cannot restore transformed attention after model unload")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        for layer_index, attention in self._original_transform_attentions.items():
            layers[layer_index].self_attn = attention
        self._original_transform_attentions.clear()

    def make_cache(self) -> Any:
        if self.model is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        cache_module = import_module("mlx_lm.models.cache", "mlx-lm")
        return cache_module.make_prompt_cache(self.model)

    def attention_geometry(self) -> dict[str, int]:
        """Return the reviewed attention dimensions needed by Phase 2 mapping."""

        if self.model is None or self._layers_path is None:
            raise RuntimeUnavailableError("Adapter is not loaded")
        parent, attribute = self._layer_parent_and_attribute()
        layers = list(getattr(parent, attribute))
        attention = layers[0].self_attn
        return {
            "layers": len(layers),
            "attention_heads": int(attention.n_heads),
            "kv_heads": int(attention.n_kv_heads),
            "hidden_width": self.architecture().hidden_width,
        }

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
        path, layers_value = resolve_attribute(self.model, ("model.layers", "layers"))
        layers = list(layers_value)
        if not layers:
            raise RuntimeUnavailableError("Qwen decoder layer collection is empty")
        config = getattr(self.model, "args", getattr(self.model, "config", None))
        hidden_width = int(getattr(config, "hidden_size", 0))
        vocabulary_size = int(getattr(config, "vocab_size", 0))
        if hidden_width <= 0 or vocabulary_size <= 0:
            raise RuntimeUnavailableError("Qwen configuration lacks hidden_size or vocab_size")
        count = len(layers)
        layer_info = tuple(
            LayerInfo(
                index=index,
                normalized_depth=index / max(1, count - 1),
                layer_type=type(layer).__name__,
                attention_kind="unknown",
            )
            for index, layer in enumerate(layers)
        )
        return (
            ArchitectureInfo(
                decoder_layer_path=path,
                hidden_width=hidden_width,
                vocabulary_size=vocabulary_size,
                layers=layer_info,
                final_norm_path="model.norm",
                output_head_path="lm_head",
                cache_type="mlx_lm.models.cache.make_prompt_cache",
                cache_count=0,
                has_per_layer_inputs=False,
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


def _dequantized_linear_weight(mx: Any, projection: Any) -> Any:
    """Return an ordinary output-by-input matrix for dense or quantized linear."""

    scales = getattr(projection, "scales", None)
    if scales is None:
        weight = projection.weight.astype(mx.float32)
    else:
        weight = mx.dequantize(
            projection.weight,
            scales,
            getattr(projection, "biases", None),
            group_size=int(projection.group_size),
            bits=int(projection.bits),
            mode=str(projection.mode),
            dtype=mx.float32,
        )
    mx.eval(weight)
    return weight
