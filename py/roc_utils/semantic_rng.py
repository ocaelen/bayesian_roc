"""Deterministic random streams for the scientific experiment runners."""

from __future__ import annotations

import numpy as np

if not __debug__:
    raise RuntimeError(
        "semantic_rng relies on runtime assertions; do not use python -O."
    )


RNG_ALGORITHM = "PCG64"
_MAX_KEY_COMPONENT = 2**32 - 1


def _validate_key_component(name: str, value: int) -> int:
    """Return one validated non-negative 32-bit integer key component."""

    if not isinstance(value, (int, np.integer)) or isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if not 0 <= value <= _MAX_KEY_COMPONENT:
        raise ValueError(
            f"{name} must lie between 0 and {_MAX_KEY_COMPONENT}."
        )
    return value


def normalize_semantic_key(*semantic_key: int) -> tuple[int, ...]:
    """Validate a non-empty semantic key used to identify one RNG stream."""

    if not semantic_key:
        raise ValueError("A semantic RNG key must not be empty.")
    normalized = tuple(
        _validate_key_component(f"semantic_key[{index}]", component)
        for index, component in enumerate(semantic_key)
    )
    assert len(normalized) == len(semantic_key)
    return normalized


def semantic_stream_id(master_seed: int, *semantic_key: int) -> str:
    """Return a stable identifier for one complete PCG64 stream."""

    master_seed = _validate_key_component("master_seed", master_seed)
    normalized = normalize_semantic_key(*semantic_key)
    stream_id = ":".join(
        (
            RNG_ALGORITHM,
            str(master_seed),
            *(str(component) for component in normalized),
        )
    )
    assert stream_id
    stream_parts = stream_id.split(":")
    assert stream_parts[0] == RNG_ALGORITHM
    assert int(stream_parts[1]) == master_seed
    assert tuple(int(component) for component in stream_parts[2:]) == normalized
    return stream_id


def make_semantic_rng(
    master_seed: int,
    *semantic_key: int,
) -> np.random.Generator:
    """
    Construct one order-independent NumPy PCG64 stream.

    The master seed selects a complete experiment run. The semantic key must
    identify the scientific role of one stream, such as data generation or
    posterior sampling, together with its condition and replication. Direct
    ``SeedSequence`` keys make the result independent of loop order and avoid
    Python's process-dependent ``hash`` function.
    """

    master_seed = _validate_key_component("master_seed", master_seed)
    normalized_key = normalize_semantic_key(*semantic_key)
    seed_sequence = np.random.SeedSequence(
        entropy=master_seed,
        spawn_key=normalized_key,
    )
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    assert isinstance(rng, np.random.Generator)
    assert isinstance(rng.bit_generator, np.random.PCG64)
    return rng
