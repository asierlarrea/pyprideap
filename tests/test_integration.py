"""End-to-end tests using the public API."""

import pyap


def test_read_validate_stats_olink(olink_csv_path):
    dataset = pyap.read(olink_csv_path)
    results = pyap.validate(dataset)
    stats = pyap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)


def test_read_validate_stats_somascan(somascan_adat_path):
    dataset = pyap.read(somascan_adat_path)
    results = pyap.validate(dataset)
    stats = pyap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)


def test_read_validate_stats_parquet(olink_parquet_path):
    dataset = pyap.read(olink_parquet_path)
    results = pyap.validate(dataset)
    stats = pyap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)


def test_read_validate_stats_somascan_csv(somascan_csv_path):
    dataset = pyap.read(somascan_csv_path)
    results = pyap.validate(dataset)
    stats = pyap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)
