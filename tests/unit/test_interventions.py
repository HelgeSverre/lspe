import numpy as np

from lspe.interventions import InterventionController, rms_scaled_additive, spherical_rotation


def test_spherical_rotation_preserves_norm() -> None:
    activation = np.array([[3.0, 4.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    result = spherical_rotation(activation, np.array([0.2, 0.7, -0.3]), theta=0.3)
    np.testing.assert_allclose(
        np.linalg.vector_norm(result.activation, axis=-1),
        np.linalg.vector_norm(activation, axis=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_spherical_rotation_handles_collinear_direction() -> None:
    activation = np.array([[3.0, 4.0]], dtype=np.float32)
    result = spherical_rotation(activation, np.array([3.0, 4.0]), theta=0.3)
    np.testing.assert_allclose(
        np.linalg.vector_norm(result.activation),
        np.linalg.vector_norm(activation),
        rtol=1e-6,
        atol=1e-6,
    )


def test_additive_zero_dose_identity() -> None:
    activation = np.array([[1.0, 2.0]], dtype=np.float32)
    result = rms_scaled_additive(activation, np.array([3.0, 4.0]), alpha=0.0)
    np.testing.assert_array_equal(result.activation, activation)


def test_direction_reproducibility() -> None:
    kwargs = dict(
        master_seed=3,
        run_id="run",
        prompt_id="prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({2}),
        dose=0.1,
    )
    activation = np.ones((1, 4), dtype=np.float32)
    first = InterventionController(**kwargs).apply_post_layer(2, activation, 1)
    second = InterventionController(**kwargs).apply_post_layer(2, activation, 1)
    np.testing.assert_array_equal(first, second)


def test_direction_changes_with_seed() -> None:
    kwargs = dict(
        run_id="run",
        prompt_id="prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({2}),
        dose=0.1,
    )
    activation = np.ones((1, 4), dtype=np.float32)
    first = InterventionController(master_seed=3, **kwargs).apply_post_layer(2, activation, 1)
    second = InterventionController(master_seed=4, **kwargs).apply_post_layer(2, activation, 1)
    assert not np.array_equal(first, second)


def test_white_noise_uses_new_direction_per_token() -> None:
    controller = InterventionController(
        master_seed=3,
        run_id="run",
        prompt_id="prompt",
        generation_index=0,
        condition_id="white",
        selected_layers=frozenset({2}),
        dose=0.1,
        mode="white_per_token",
    )
    activation = np.ones((1, 4), dtype=np.float32)
    first = controller.apply_post_layer(2, activation, 1)
    second = controller.apply_post_layer(2, activation, 2)
    assert not np.array_equal(first, second)


def test_coherent_mode_reuses_direction() -> None:
    controller = InterventionController(
        master_seed=3,
        run_id="run",
        prompt_id="prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({2}),
        dose=0.1,
    )
    activation = np.ones((1, 4), dtype=np.float32)
    first = controller.apply_post_layer(2, activation, 1)
    second = controller.apply_post_layer(2, activation, 2)
    np.testing.assert_array_equal(first, second)


def test_direction_fingerprints_are_stable_without_storing_vectors() -> None:
    controller = InterventionController(
        master_seed=3,
        run_id="run",
        prompt_id="prompt",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset({2}),
        dose=0.1,
    )
    controller.apply_post_layer(2, np.ones((1, 4), dtype=np.float32), 1)
    fingerprints = controller.direction_fingerprints()
    assert fingerprints[0]["layer_index"] == 2
    assert len(str(fingerprints[0]["direction_sha256"])) == 64
    assert fingerprints == controller.direction_fingerprints()
