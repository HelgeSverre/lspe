from lspe.rng import derive_seed


def test_rng_domains_are_independent() -> None:
    shared = (721984, "prompt-1", 0)
    assert derive_seed(shared[0], "prompt-order", *shared[1:]) != derive_seed(
        shared[0], "condition-order", *shared[1:]
    )


def test_rng_reproducibility() -> None:
    assert derive_seed(12, "sampling-token", "p", 1, 2) == derive_seed(
        12, "sampling-token", "p", 1, 2
    )
