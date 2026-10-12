"""
Download and prepare one real dataset replication from ``main.tex``.

The registry records the dataset definitions documented in the manuscript.
Each source is pinned by OpenML ID, version, file ID, and MD5.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder

if not __debug__:
    raise RuntimeError(
        "real_datasets relies on runtime assertions; do not use python -O."
    )


DEFAULT_REAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MISSING_STRINGS = {"", "?", "na", "n/a", "nan", "none", "null", "missing"}


@dataclass(frozen=True)
class _RealDatasetSpec:
    name: str
    slug: str
    openml_data_id: int
    openml_version: int
    openml_file_id: int
    md5_checksum: str
    target_column: str
    positive_values: tuple[str, ...]
    negative_values: tuple[str, ...]
    positive_threshold: float | None
    drop_features: tuple[str, ...]
    categorical_overrides: tuple[str, ...]
    expected_raw_rows: int
    expected_complete_rows: int
    expected_positive: int
    expected_feature_count: int
    expected_categorical_features: tuple[str, ...]

    @property
    def download_url(self) -> str:
        return (
            "https://www.openml.org/data/v1/download/"
            f"{self.openml_file_id}"
        )


@dataclass(frozen=True)
class RealDatasetMetadata:
    """Pinned source, binary task, and counts available without downloading."""

    name: str
    slug: str
    openml_data_id: int
    openml_version: int
    openml_file_id: int
    md5_checksum: str
    n_source_rows: int
    n_complete_rows: int
    n_removed_rows: int
    n_positive: int
    n_negative: int
    target_column: str
    positive_values: tuple[str, ...]
    negative_values: tuple[str, ...]
    positive_threshold: float | None
    drop_features: tuple[str, ...]
    categorical_overrides: tuple[str, ...]


@dataclass(frozen=True)
class LoadedRealDataset:
    """Immutable complete-case source data shared across replications."""

    name: str
    slug: str
    source: str
    source_path: Path
    openml_data_id: int
    openml_version: int
    openml_file_id: int
    md5_checksum: str
    n_source_rows: int
    n_complete_rows: int
    n_removed_rows: int
    feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    X_raw: np.ndarray
    y: np.ndarray
    source_row_indices: np.ndarray


@dataclass(frozen=True)
class PreparedRealDataset:
    """Numeric train/test/external arrays for one real-data replication."""

    name: str
    slug: str
    source: str
    source_path: Path
    openml_data_id: int
    openml_version: int
    openml_file_id: int
    md5_checksum: str
    n_source_rows: int
    n_complete_rows: int
    n_removed_rows: int
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_external: np.ndarray
    y_external: np.ndarray
    train_row_indices: np.ndarray
    test_row_indices: np.ndarray
    external_row_indices: np.ndarray


_REAL_DATASET_SPECS = (
    _RealDatasetSpec(
        name="Abalone",
        slug="abalone",
        openml_data_id=183,
        openml_version=1,
        openml_file_id=3620,
        md5_checksum="a9555b1cfe39eb55df2c520f09172a96",
        target_column="Class_number_of_rings",
        positive_values=(),
        negative_values=(),
        positive_threshold=10.0,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=4177,
        expected_complete_rows=4177,
        expected_positive=2081,
        expected_feature_count=8,
        expected_categorical_features=("Sex",),
    ),
    _RealDatasetSpec(
        name="Adult",
        slug="adult",
        openml_data_id=1590,
        openml_version=2,
        openml_file_id=1595261,
        md5_checksum="bb6510925e5d4b23d136715febb2cdf5",
        target_column="class",
        positive_values=(">50K",),
        negative_values=("<=50K",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=48842,
        expected_complete_rows=45222,
        expected_positive=11208,
        expected_feature_count=14,
        expected_categorical_features=(
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ),
    ),
    _RealDatasetSpec(
        name="Alzheimer",
        slug="alzheimer",
        openml_data_id=46849,
        openml_version=2,
        openml_file_id=22124412,
        md5_checksum="83e3bd9a41795e8e6916b0e511416c2c",
        target_column="class",
        positive_values=("P",),
        negative_values=("H",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=174,
        expected_complete_rows=174,
        expected_positive=89,
        expected_feature_count=450,
        expected_categorical_features=(),
    ),
    _RealDatasetSpec(
        name="Banknote Authentication",
        slug="banknote_authentication",
        openml_data_id=1462,
        openml_version=1,
        openml_file_id=1586223,
        md5_checksum="baa2dc5b745775a943ebeb9c276401f8",
        target_column="Class",
        positive_values=("2",),
        negative_values=("1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=1372,
        expected_complete_rows=1372,
        expected_positive=610,
        expected_feature_count=4,
        expected_categorical_features=(),
    ),
    _RealDatasetSpec(
        name="Diabetes",
        slug="diabetes",
        openml_data_id=37,
        openml_version=1,
        openml_file_id=37,
        md5_checksum="3cbaa3e54586aa88cf6aacb4033e4470",
        target_column="class",
        positive_values=("tested_positive",),
        negative_values=("tested_negative",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=768,
        expected_complete_rows=768,
        expected_positive=268,
        expected_feature_count=8,
        expected_categorical_features=(),
    ),
    _RealDatasetSpec(
        name="Fertility",
        slug="fertility",
        openml_data_id=1473,
        openml_version=1,
        openml_file_id=1587930,
        md5_checksum="6bfc3c6c5c7cd86324ce16e63a623bef",
        target_column="Class",
        positive_values=("2",),
        negative_values=("1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(
            "V1",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
        ),
        expected_raw_rows=100,
        expected_complete_rows=100,
        expected_positive=12,
        expected_feature_count=9,
        expected_categorical_features=(
            "V1",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
        ),
    ),
    _RealDatasetSpec(
        name="Haberman",
        slug="haberman",
        openml_data_id=43,
        openml_version=1,
        openml_file_id=43,
        md5_checksum="fe063732e696d8e7aa79746032f0191b",
        target_column="Survival_status",
        positive_values=("2",),
        negative_values=("1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=306,
        expected_complete_rows=306,
        expected_positive=81,
        expected_feature_count=3,
        expected_categorical_features=("Patients_year_of_operation",),
    ),
    _RealDatasetSpec(
        name="Mammographic",
        slug="mammographic",
        openml_data_id=45557,
        openml_version=2,
        openml_file_id=22116525,
        md5_checksum="16eda295e59d814dde5cadbb97330117",
        target_column="Severity",
        positive_values=("1",),
        negative_values=("0",),
        positive_threshold=None,
        drop_features=("BI-RADS",),
        categorical_overrides=(),
        expected_raw_rows=961,
        expected_complete_rows=830,
        expected_positive=403,
        expected_feature_count=4,
        expected_categorical_features=("Shape", "Margin"),
    ),
    _RealDatasetSpec(
        name="Sonar",
        slug="sonar",
        openml_data_id=40,
        openml_version=1,
        openml_file_id=40,
        md5_checksum="3ab630fbbfe25ab48b9bb47ce5759203",
        target_column="Class",
        positive_values=("Mine",),
        negative_values=("Rock",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=208,
        expected_complete_rows=208,
        expected_positive=111,
        expected_feature_count=60,
        expected_categorical_features=(),
    ),
    _RealDatasetSpec(
        name="Titanic",
        slug="titanic",
        openml_data_id=40704,
        openml_version=2,
        openml_file_id=4965305,
        md5_checksum="08416114dd85d0ebd932fcb1d87650c1",
        target_column="class",
        positive_values=("1",),
        negative_values=("-1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=("Class", "Age", "Sex"),
        expected_raw_rows=2201,
        expected_complete_rows=2201,
        expected_positive=711,
        expected_feature_count=3,
        expected_categorical_features=("Class", "Age", "Sex"),
    ),
    _RealDatasetSpec(
        name="Transfusion",
        slug="transfusion",
        openml_data_id=1464,
        openml_version=1,
        openml_file_id=1586225,
        md5_checksum="c3242468edab8c2e7a907674122dc851",
        target_column="Class",
        positive_values=("2",),
        negative_values=("1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=748,
        expected_complete_rows=748,
        expected_positive=178,
        expected_feature_count=4,
        expected_categorical_features=(),
    ),
    _RealDatasetSpec(
        name="Contraceptive Method Choice",
        slug="contraceptive_method_choice",
        openml_data_id=23,
        openml_version=1,
        openml_file_id=23,
        md5_checksum="3149646ecff276abac3e892d1556655f",
        target_column="Contraceptive_method_used",
        positive_values=("2", "3"),
        negative_values=("1",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=1473,
        expected_complete_rows=1473,
        expected_positive=844,
        expected_feature_count=9,
        expected_categorical_features=(
            "Wifes_education",
            "Husbands_education",
            "Wifes_religion",
            "Wifes_now_working%3F",
            "Husbands_occupation",
            "Standard-of-living_index",
            "Media_exposure",
        ),
    ),
)

REAL_DATASET_NAMES = tuple(spec.name for spec in _REAL_DATASET_SPECS)


def _dataset_key(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


_SPECS_BY_KEY = {
    key: spec
    for spec in _REAL_DATASET_SPECS
    for key in (_dataset_key(spec.name), _dataset_key(spec.slug))
}

assert len(_REAL_DATASET_SPECS) == 12
assert len(set(REAL_DATASET_NAMES)) == 12
assert len({spec.openml_data_id for spec in _REAL_DATASET_SPECS}) == 12


def real_dataset_metadata(dataset_name: str) -> RealDatasetMetadata:
    """Return the pinned source identity and complete-case class counts."""

    assert isinstance(dataset_name, str), "dataset_name must be a string"
    assert _dataset_key(dataset_name) in _SPECS_BY_KEY, (
        f"Unknown dataset {dataset_name!r}; choose one of {REAL_DATASET_NAMES}"
    )
    spec = _SPECS_BY_KEY[_dataset_key(dataset_name)]
    metadata = RealDatasetMetadata(
        name=spec.name,
        slug=spec.slug,
        openml_data_id=spec.openml_data_id,
        openml_version=spec.openml_version,
        openml_file_id=spec.openml_file_id,
        md5_checksum=spec.md5_checksum,
        n_source_rows=spec.expected_raw_rows,
        n_complete_rows=spec.expected_complete_rows,
        n_removed_rows=spec.expected_raw_rows - spec.expected_complete_rows,
        n_positive=spec.expected_positive,
        n_negative=spec.expected_complete_rows - spec.expected_positive,
        target_column=spec.target_column,
        positive_values=spec.positive_values,
        negative_values=spec.negative_values,
        positive_threshold=spec.positive_threshold,
        drop_features=spec.drop_features,
        categorical_overrides=spec.categorical_overrides,
    )
    assert metadata.n_complete_rows == (
        metadata.n_positive + metadata.n_negative
    )
    assert metadata.n_removed_rows >= 0
    assert metadata.n_positive >= 5
    assert metadata.n_negative >= 5
    assert metadata.target_column
    if metadata.positive_threshold is None:
        assert metadata.positive_values and metadata.negative_values
        assert set(metadata.positive_values).isdisjoint(metadata.negative_values)
    else:
        assert np.isfinite(metadata.positive_threshold)
        assert not metadata.positive_values and not metadata.negative_values
    return metadata


def load_real_dataset(
    dataset_name: str,
    *,
    data_dir: str | Path = DEFAULT_REAL_DATA_DIR,
) -> LoadedRealDataset:
    """
    Download and parse one immutable complete-case real dataset.

    A real-data experiment should call this function once per dataset, then
    pass the returned object to ``split_and_encode_real_dataset`` for every
    replication. Source acquisition, checksum verification, parsing, missing
    row removal, and target recoding are therefore performed only once.
    """

    assert isinstance(dataset_name, str), "dataset_name must be a string"
    assert _dataset_key(dataset_name) in _SPECS_BY_KEY, (
        f"Unknown dataset {dataset_name!r}; choose one of {REAL_DATASET_NAMES}"
    )

    spec = _SPECS_BY_KEY[_dataset_key(dataset_name)]
    data_dir = Path(data_dir)
    if data_dir.exists():
        assert data_dir.is_dir(), f"data_dir is not a directory: {data_dir}"
    else:
        data_dir.mkdir(parents=True)
    assert data_dir.is_dir(), f"Could not create data directory: {data_dir}"

    source_path = data_dir / f"{spec.slug}.arff"
    _ensure_source_file(spec, source_path)
    (
        X_raw,
        y,
        source_row_indices,
        feature_names,
        categorical_feature_names,
    ) = _load_complete_dataset(spec, source_path)

    X_raw.setflags(write=False)
    y.setflags(write=False)
    source_row_indices.setflags(write=False)

    dataset = LoadedRealDataset(
        name=spec.name,
        slug=spec.slug,
        source=(
            f"OpenML data_id={spec.openml_data_id}, "
            f"version={spec.openml_version}"
        ),
        source_path=source_path,
        openml_data_id=spec.openml_data_id,
        openml_version=spec.openml_version,
        openml_file_id=spec.openml_file_id,
        md5_checksum=spec.md5_checksum,
        n_source_rows=spec.expected_raw_rows,
        n_complete_rows=int(y.size),
        n_removed_rows=spec.expected_raw_rows - int(y.size),
        feature_names=feature_names,
        categorical_feature_names=categorical_feature_names,
        X_raw=X_raw,
        y=y,
        source_row_indices=source_row_indices,
    )
    _assert_loaded_dataset(dataset)
    return dataset


def _assert_split_rng(rng: np.random.Generator) -> None:
    """Assert the minimal RNG interface required by the classwise split."""

    assert callable(getattr(rng, "permutation", None)), (
        "rng must provide a callable permutation method"
    )


def split_and_encode_real_dataset(
    dataset: LoadedRealDataset,
    *,
    rng: np.random.Generator,
) -> PreparedRealDataset:
    """
    Split and training-encode one replication of a loaded real dataset.

    Missing rows are removed before an exact within-class split. For each
    class, ``floor(n_class / 5)`` rows go to test, the same number go to the
    external set, and all remaining rows go to training. Categorical encoding
    is fitted on training rows only and reused unchanged for both held-out
    sets. Numerical features are left on their original scales.
    """

    assert isinstance(dataset, LoadedRealDataset)
    _assert_loaded_dataset(dataset)
    _assert_split_rng(rng)

    train_indices, test_indices, external_indices = _split_within_classes(
        dataset.y,
        rng,
    )
    (
        X_train,
        X_test,
        X_external,
        encoded_feature_names,
        numeric_feature_names,
    ) = _encode_from_training_only(
        dataset.X_raw,
        dataset.feature_names,
        dataset.categorical_feature_names,
        train_indices,
        test_indices,
        external_indices,
    )

    result = PreparedRealDataset(
        name=dataset.name,
        slug=dataset.slug,
        source=dataset.source,
        source_path=dataset.source_path,
        openml_data_id=dataset.openml_data_id,
        openml_version=dataset.openml_version,
        openml_file_id=dataset.openml_file_id,
        md5_checksum=dataset.md5_checksum,
        n_source_rows=dataset.n_source_rows,
        n_complete_rows=dataset.n_complete_rows,
        n_removed_rows=dataset.n_removed_rows,
        numeric_feature_names=numeric_feature_names,
        categorical_feature_names=dataset.categorical_feature_names,
        feature_names=encoded_feature_names,
        X_train=X_train,
        y_train=dataset.y[train_indices].copy(),
        X_test=X_test,
        y_test=dataset.y[test_indices].copy(),
        X_external=X_external,
        y_external=dataset.y[external_indices].copy(),
        train_row_indices=dataset.source_row_indices[train_indices].copy(),
        test_row_indices=dataset.source_row_indices[test_indices].copy(),
        external_row_indices=(
            dataset.source_row_indices[external_indices].copy()
        ),
    )
    _assert_prepared_dataset(
        result,
        dataset.y,
        dataset.source_row_indices,
    )
    return result


def prepare_real_dataset(
    dataset_name: str,
    *,
    rng: np.random.Generator,
    data_dir: str | Path = DEFAULT_REAL_DATA_DIR,
) -> PreparedRealDataset:
    """Load, split, and encode one real dataset through the one-call API."""

    _assert_split_rng(rng)
    dataset = load_real_dataset(dataset_name, data_dir=data_dir)
    return split_and_encode_real_dataset(dataset, rng=rng)


def _ensure_source_file(spec: _RealDatasetSpec, path: Path) -> None:
    if path.exists():
        assert path.is_file()
        actual_checksum = md5(path.read_bytes()).hexdigest()
        if actual_checksum != spec.md5_checksum:
            raise RuntimeError(
                f"Cached file {path} has MD5 {actual_checksum}, expected "
                f"{spec.md5_checksum}. Remove or replace the corrupted file."
            )
        return

    partial_path = path.with_suffix(".arff.part")
    try:
        request = Request(
            spec.download_url,
            headers={"User-Agent": "bayesian-roc-scientific-experiment"},
        )
        with urlopen(request, timeout=60) as response:
            content = response.read()
        actual_checksum = md5(content).hexdigest()
        if actual_checksum != spec.md5_checksum:
            raise RuntimeError(
                f"Downloaded MD5 {actual_checksum}, expected "
                f"{spec.md5_checksum}."
            )
        partial_path.write_bytes(content)
        partial_path.replace(path)
    except Exception as error:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {spec.name!r} from {spec.download_url} "
            f"into {path}: {error}"
        ) from error

    assert path.is_file()
    assert md5(path.read_bytes()).hexdigest() == spec.md5_checksum


def _python_value(value: object) -> object:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: object) -> bool:
    value = _python_value(value)
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _MISSING_STRINGS
    return False


def _normalize_label(value: object) -> str:
    text = str(_python_value(value)).strip().strip("\"'")
    return text.rstrip(".").casefold()


def _categorical_text(value: object) -> str:
    value = _python_value(value)
    if isinstance(value, float):
        assert np.isfinite(value)
        return np.format_float_positional(value, trim="-")
    return str(value).strip()


def _encode_target(
    raw_target: list[object],
    spec: _RealDatasetSpec,
) -> np.ndarray:
    assert raw_target

    if spec.positive_threshold is not None:
        numeric_target = np.asarray(raw_target, dtype=float)
        assert np.all(np.isfinite(numeric_target))
        y = (numeric_target >= spec.positive_threshold).astype(np.int64)
    else:
        labels = [_normalize_label(value) for value in raw_target]
        positive = {_normalize_label(value) for value in spec.positive_values}
        negative = {_normalize_label(value) for value in spec.negative_values}
        assert positive
        assert negative
        assert positive.isdisjoint(negative)
        assert set(labels) == positive | negative
        y = np.asarray([int(label in positive) for label in labels], dtype=np.int64)

    assert y.ndim == 1
    assert set(np.unique(y)) == {0, 1}
    return y


def _load_complete_dataset(
    spec: _RealDatasetSpec,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    try:
        records, metadata = arff.loadarff(path)
    except Exception as error:
        raise RuntimeError(f"Could not parse ARFF file {path}: {error}") from error

    all_columns = tuple(metadata.names())
    all_types = dict(zip(all_columns, metadata.types()))
    assert spec.target_column in all_columns
    assert all(feature in all_columns for feature in spec.drop_features)
    assert len(records) == spec.expected_raw_rows

    feature_names = tuple(
        column
        for column in all_columns
        if column != spec.target_column and column not in spec.drop_features
    )
    assert len(feature_names) == spec.expected_feature_count
    assert len(set(feature_names)) == len(feature_names)

    categorical_set = {
        name for name in feature_names if all_types[name] == "nominal"
    }
    categorical_set.update(spec.categorical_overrides)
    assert categorical_set.issubset(feature_names)
    categorical_feature_names = tuple(
        name for name in feature_names if name in categorical_set
    )
    assert categorical_feature_names == spec.expected_categorical_features

    rows = []
    raw_target = []
    source_row_indices = []
    for source_index, record in enumerate(records):
        values = {
            column: _python_value(record[column])
            for column in all_columns
        }
        if any(_is_missing(value) for value in values.values()):
            continue

        row = []
        for feature_name in feature_names:
            value = values[feature_name]
            if feature_name in categorical_set:
                row.append(_categorical_text(value))
            else:
                numeric_value = float(value)
                assert np.isfinite(numeric_value)
                row.append(numeric_value)

        rows.append(row)
        raw_target.append(values[spec.target_column])
        source_row_indices.append(source_index)

    X_raw = np.asarray(rows, dtype=object)
    y = _encode_target(raw_target, spec)
    source_row_indices = np.asarray(source_row_indices, dtype=np.int64)

    assert X_raw.shape == (spec.expected_complete_rows, len(feature_names))
    assert y.shape == (spec.expected_complete_rows,)
    assert source_row_indices.shape == (spec.expected_complete_rows,)
    assert int(y.sum()) == spec.expected_positive
    assert int(y.size - y.sum()) == (
        spec.expected_complete_rows - spec.expected_positive
    )

    return (
        X_raw,
        y,
        source_row_indices,
        feature_names,
        categorical_feature_names,
    )


def _split_within_classes(
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    external_parts = []

    for label in (0, 1):
        class_indices = np.flatnonzero(y == label)
        shuffled = np.asarray(rng.permutation(class_indices), dtype=np.int64)
        assert shuffled.shape == class_indices.shape
        assert np.array_equal(np.sort(shuffled), class_indices)

        held_out_count = int(class_indices.size // 5)
        assert held_out_count >= 1
        test_parts.append(shuffled[:held_out_count])
        external_parts.append(shuffled[held_out_count : 2 * held_out_count])
        train_parts.append(shuffled[2 * held_out_count :])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    external_indices = np.concatenate(external_parts)

    all_indices = np.concatenate(
        (train_indices, test_indices, external_indices)
    )
    assert np.unique(all_indices).size == y.size
    assert np.array_equal(np.sort(all_indices), np.arange(y.size))

    for label in (0, 1):
        expected_held_out = int(np.count_nonzero(y == label) // 5)
        assert np.count_nonzero(y[test_indices] == label) == expected_held_out
        assert np.count_nonzero(y[external_indices] == label) == expected_held_out
        assert np.count_nonzero(y[train_indices] == label) == (
            np.count_nonzero(y == label) - 2 * expected_held_out
        )

    return train_indices, test_indices, external_indices


def _encode_from_training_only(
    X_raw: np.ndarray,
    feature_names: tuple[str, ...],
    categorical_feature_names: tuple[str, ...],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    external_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    categorical_set = set(categorical_feature_names)
    numeric_feature_names = tuple(
        name for name in feature_names if name not in categorical_set
    )
    numeric_indices = [
        index
        for index, name in enumerate(feature_names)
        if name not in categorical_set
    ]
    categorical_indices = [
        index
        for index, name in enumerate(feature_names)
        if name in categorical_set
    ]

    def numeric_part(indices: np.ndarray) -> np.ndarray:
        if not numeric_indices:
            return np.empty((indices.size, 0), dtype=float)
        values = np.asarray(
            X_raw[np.ix_(indices, numeric_indices)],
            dtype=float,
        )
        assert np.all(np.isfinite(values))
        return values

    X_train_numeric = numeric_part(train_indices)
    X_test_numeric = numeric_part(test_indices)
    X_external_numeric = numeric_part(external_indices)

    if categorical_indices:
        train_categories = np.asarray(
            X_raw[np.ix_(train_indices, categorical_indices)],
            dtype=str,
        )
        test_categories = np.asarray(
            X_raw[np.ix_(test_indices, categorical_indices)],
            dtype=str,
        )
        external_categories = np.asarray(
            X_raw[np.ix_(external_indices, categorical_indices)],
            dtype=str,
        )

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=float,
        )
        X_train_categorical = encoder.fit_transform(train_categories)
        X_test_categorical = encoder.transform(test_categories)
        X_external_categorical = encoder.transform(external_categories)
        encoded_categorical_names = tuple(
            encoder.get_feature_names_out(categorical_feature_names)
        )

        for column, categories in enumerate(encoder.categories_):
            assert set(categories) == set(train_categories[:, column])
    else:
        X_train_categorical = np.empty((train_indices.size, 0), dtype=float)
        X_test_categorical = np.empty((test_indices.size, 0), dtype=float)
        X_external_categorical = np.empty(
            (external_indices.size, 0),
            dtype=float,
        )
        encoded_categorical_names = ()

    X_train = np.hstack((X_train_numeric, X_train_categorical))
    X_test = np.hstack((X_test_numeric, X_test_categorical))
    X_external = np.hstack((X_external_numeric, X_external_categorical))
    encoded_feature_names = numeric_feature_names + encoded_categorical_names

    assert len(encoded_feature_names) == X_train.shape[1]
    assert len(set(encoded_feature_names)) == len(encoded_feature_names)
    return (
        X_train,
        X_test,
        X_external,
        encoded_feature_names,
        numeric_feature_names,
    )


def _assert_loaded_dataset(dataset: LoadedRealDataset) -> None:
    """Assert the immutable source-data contract used across replications."""

    assert isinstance(dataset, LoadedRealDataset)
    assert dataset.name
    assert dataset.slug
    assert dataset.source_path.is_file()
    assert dataset.n_source_rows > 0
    assert 0 < dataset.n_complete_rows <= dataset.n_source_rows
    assert dataset.n_removed_rows == (
        dataset.n_source_rows - dataset.n_complete_rows
    )
    assert len(dataset.feature_names) > 0
    assert len(set(dataset.feature_names)) == len(dataset.feature_names)
    assert set(dataset.categorical_feature_names).issubset(
        dataset.feature_names
    )

    assert dataset.X_raw.shape == (
        dataset.n_complete_rows,
        len(dataset.feature_names),
    )
    assert dataset.y.shape == (dataset.n_complete_rows,)
    assert dataset.y.dtype == np.int64
    assert set(np.unique(dataset.y)) == {0, 1}
    assert dataset.source_row_indices.shape == (dataset.n_complete_rows,)
    assert dataset.source_row_indices.dtype == np.int64
    assert np.all(np.diff(dataset.source_row_indices) > 0)
    assert dataset.source_row_indices[0] >= 0
    assert dataset.source_row_indices[-1] < dataset.n_source_rows

    assert not dataset.X_raw.flags.writeable
    assert not dataset.y.flags.writeable
    assert not dataset.source_row_indices.flags.writeable


def _assert_prepared_dataset(
    dataset: PreparedRealDataset,
    complete_y: np.ndarray,
    complete_source_indices: np.ndarray,
) -> None:
    assert dataset.n_complete_rows == complete_y.size
    assert dataset.n_removed_rows == (
        dataset.n_source_rows - dataset.n_complete_rows
    )
    assert dataset.n_removed_rows >= 0
    assert dataset.source_path.is_file()

    for X, y, row_indices in (
        (dataset.X_train, dataset.y_train, dataset.train_row_indices),
        (dataset.X_test, dataset.y_test, dataset.test_row_indices),
        (
            dataset.X_external,
            dataset.y_external,
            dataset.external_row_indices,
        ),
    ):
        assert X.ndim == 2
        assert y.ndim == 1
        assert row_indices.ndim == 1
        assert X.shape[0] == y.size == row_indices.size
        assert X.shape[1] == len(dataset.feature_names)
        assert np.all(np.isfinite(X))
        assert y.dtype == np.int64
        assert set(np.unique(y)) == {0, 1}

    all_source_indices = np.concatenate(
        (
            dataset.train_row_indices,
            dataset.test_row_indices,
            dataset.external_row_indices,
        )
    )
    assert np.unique(all_source_indices).size == dataset.n_complete_rows
    assert np.array_equal(
        np.sort(all_source_indices),
        complete_source_indices,
    )

    for label in (0, 1):
        total = int(np.count_nonzero(complete_y == label))
        expected_held_out = total // 5
        assert np.count_nonzero(dataset.y_test == label) == expected_held_out
        assert (
            np.count_nonzero(dataset.y_external == label)
            == expected_held_out
        )
        assert np.count_nonzero(dataset.y_train == label) == (
            total - 2 * expected_held_out
        )
