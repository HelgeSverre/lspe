import mlx.core as mx
import numpy as np

from lspe.interventions.controller import InterventionController


def test_mlx_spherical_intervention_preserves_norm() -> None:
    controller = InterventionController(
        master_seed=7,
        run_id="test-run",
        prompt_id="test-prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({0}),
        dose=0.1,
        decode_start_token=0,
    )
    activation = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    altered = controller.apply_post_layer_mlx(0, activation, 0)
    mx.eval(altered)
    np.testing.assert_allclose(
        np.asarray(mx.linalg.norm(activation, axis=-1)),
        np.asarray(mx.linalg.norm(altered, axis=-1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mlx_additive_intervention_preserves_rms() -> None:
    controller = InterventionController(
        master_seed=7,
        run_id="test-run",
        prompt_id="test-prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({0}),
        dose=0.1,
        kernel="rms_scaled_additive",
        decode_start_token=0,
    )
    activation = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    altered = controller.apply_post_layer_mlx(0, activation, 0)
    mx.eval(altered)
    np.testing.assert_allclose(
        np.asarray(mx.sqrt(mx.mean(activation * activation, axis=-1))),
        np.asarray(mx.sqrt(mx.mean(altered * altered, axis=-1))),
        rtol=1e-5,
        atol=1e-5,
    )
