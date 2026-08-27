"""Deterministic layer/time controller for transient interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..hashing import sha256_bytes
from ..rng import derive_seed
from .additive import rms_scaled_additive
from .spherical import spherical_rotation

DirectionMode = Literal["coherent_per_layer", "coherent_shared", "white_per_token", "zero"]
KernelName = Literal["spherical_rotation", "rms_scaled_additive"]


@dataclass(frozen=True)
class InterventionTelemetry:
    layer_index: int
    token_index: int
    active: bool
    dose: float
    norm_before: float
    norm_after: float
    near_zero_hidden_count: int
    near_zero_direction_count: int
    direction_seed: int | None


@dataclass
class InterventionController:
    """Applies a selected intervention with generation-local deterministic state."""

    master_seed: int
    run_id: str
    prompt_id: str
    generation_index: int
    condition_id: str
    selected_layers: frozenset[int]
    dose: float
    mode: DirectionMode = "coherent_per_layer"
    kernel: KernelName = "spherical_rotation"
    decode_start_token: int = 0
    epsilon: float = 1e-8
    _directions: dict[tuple[int, int | None, int], np.ndarray] = field(default_factory=dict)
    telemetry: list[InterventionTelemetry] = field(default_factory=list)

    def apply_post_layer(
        self, layer_index: int, activation: np.ndarray, token_index: int
    ) -> np.ndarray:
        """Return a transformed activation only for selected decode tokens."""

        source = np.asarray(activation)
        active = (
            layer_index in self.selected_layers
            and token_index >= self.decode_start_token
            and self.mode != "zero"
            and self.dose != 0
        )
        if not active:
            self._record(layer_index, token_index, False, source, source, 0, 0, None)
            return source
        direction, seed = self._direction(layer_index, token_index, source.shape[-1])
        if self.kernel == "spherical_rotation":
            result = spherical_rotation(source, direction, self.dose, self.epsilon)
        else:
            result = rms_scaled_additive(source, direction, self.dose, self.epsilon)
        if not np.isfinite(result.activation).all():
            raise FloatingPointError("Non-finite activation produced by intervention")
        self._record(
            layer_index,
            token_index,
            True,
            source,
            result.activation,
            result.near_zero_hidden_count,
            result.near_zero_direction_count,
            seed,
        )
        return result.activation

    def apply_post_layer_mlx(
        self, layer_index: int, activation: object, token_index: int
    ) -> object:
        """Apply the selected MLX kernel without a host round-trip or weight mutation.

        The NumPy path remains the reference implementation used by model-free
        tests. Both supported kernels perform their sensitive arithmetic in
        float32 before restoring the incoming activation dtype.
        """

        active = (
            layer_index in self.selected_layers
            and token_index >= self.decode_start_token
            and self.mode != "zero"
            and self.dose != 0
        )
        if not active:
            return activation
        from ..models.runtime import import_module

        mx = import_module("mlx.core", "mlx")
        direction, _ = self._direction(layer_index, token_index, int(activation.shape[-1]))
        original_dtype = activation.dtype
        hidden = activation.astype(mx.float32)
        random_direction = mx.array(direction).astype(mx.float32)
        if self.kernel == "spherical_rotation":
            norm = mx.sqrt(mx.sum(hidden * hidden, axis=-1, keepdims=True))
            safe_norm = mx.maximum(norm, self.epsilon)
            unit = hidden / safe_norm
            orthogonal_raw = (
                random_direction - mx.sum(random_direction * unit, axis=-1, keepdims=True) * unit
            )
            orthogonal_norm = mx.sqrt(
                mx.sum(orthogonal_raw * orthogonal_raw, axis=-1, keepdims=True)
            )
            # A real-valued Gaussian direction is almost surely non-collinear at
            # normal hidden widths. Failing closed is safer than breaking the norm invariant.
            if bool(mx.any(orthogonal_norm <= self.epsilon).item()):
                raise FloatingPointError(
                    "MLX intervention direction is collinear with a hidden state"
                )
            transformed = norm * (
                np.cos(self.dose) * unit + np.sin(self.dose) * orthogonal_raw / orthogonal_norm
            )
            output = mx.where(norm > self.epsilon, transformed, hidden)
        elif self.kernel == "rms_scaled_additive":
            rms = mx.sqrt(mx.mean(hidden * hidden, axis=-1, keepdims=True))
            direction_rms = mx.sqrt(mx.mean(random_direction * random_direction))
            normalized_direction = random_direction / mx.maximum(direction_rms, self.epsilon)
            raw = hidden + self.dose * rms * normalized_direction
            raw_rms = mx.sqrt(mx.mean(raw * raw, axis=-1, keepdims=True))
            output = raw * rms / mx.maximum(raw_rms, self.epsilon)
        else:  # Defensive guard for a malformed runtime-created controller.
            raise ValueError(f"Unsupported intervention kernel: {self.kernel}")
        output = output.astype(original_dtype)
        if not bool(mx.all(mx.isfinite(output)).item()):
            raise FloatingPointError("Non-finite MLX activation produced by intervention")
        return output

    def _direction(
        self, layer_index: int, token_index: int, hidden_width: int
    ) -> tuple[np.ndarray, int]:
        token_key: int | None = token_index if self.mode == "white_per_token" else None
        layer_key = -1 if self.mode == "coherent_shared" else layer_index
        key = (layer_key, token_key, hidden_width)
        if key not in self._directions:
            components = (
                self.run_id,
                self.prompt_id,
                self.generation_index,
                self.condition_id,
                layer_key,
                token_key,
            )
            seed = derive_seed(self.master_seed, "intervention-direction", *components)
            generator = np.random.default_rng(seed)
            self._directions[key] = generator.standard_normal(hidden_width, dtype=np.float32)
        components = (
            self.run_id,
            self.prompt_id,
            self.generation_index,
            self.condition_id,
            layer_key,
            token_key,
        )
        return self._directions[key], derive_seed(
            self.master_seed, "intervention-direction", *components
        )

    def direction_fingerprints(self) -> list[dict[str, int | str | None]]:
        """Return compact, deterministic audit records without persisting directions."""

        records: list[dict[str, int | str | None]] = []
        for (layer_key, token_key, width), direction in sorted(self._directions.items()):
            _, seed = self._direction(
                layer_key if layer_key >= 0 else next(iter(self.selected_layers)),
                token_key or 0,
                width,
            )
            records.append(
                {
                    "layer_index": layer_key,
                    "token_index": token_key,
                    "hidden_width": width,
                    "direction_seed": seed,
                    "direction_sha256": sha256_bytes(direction.tobytes()),
                }
            )
        return records

    def _record(
        self,
        layer_index: int,
        token_index: int,
        active: bool,
        before: np.ndarray,
        after: np.ndarray,
        near_zero_hidden_count: int,
        near_zero_direction_count: int,
        direction_seed: int | None,
    ) -> None:
        self.telemetry.append(
            InterventionTelemetry(
                layer_index=layer_index,
                token_index=token_index,
                active=active,
                dose=self.dose if active else 0.0,
                norm_before=float(np.linalg.vector_norm(np.asarray(before, dtype=np.float32))),
                norm_after=float(np.linalg.vector_norm(np.asarray(after, dtype=np.float32))),
                near_zero_hidden_count=near_zero_hidden_count,
                near_zero_direction_count=near_zero_direction_count,
                direction_seed=direction_seed,
            )
        )
