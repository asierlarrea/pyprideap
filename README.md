# pyap

Python library for reading, validating, and analyzing affinity proteomics datasets from the [PRIDE Affinity Archive (PAD)](https://www.ebi.ac.uk/pride/).

Supports **Olink** (Explore, Explore HT, Target) and **SomaScan** platforms.

## Installation

```bash
pip install pyap
```

Or for development:

```bash
git clone https://github.com/PRIDE-Archive/pyap.git
cd pyap
pip install -e ".[dev]"
```

## Quick Start

### Read a dataset

```python
import pyap

# Auto-detect format from file extension and content
dataset = pyap.read("olink_npx.csv")
dataset = pyap.read("raw_data.adat")
dataset = pyap.read("data.parquet")
```

### Validate against PRIDE-AP guidelines

```python
results = pyap.validate(dataset)

for r in results:
    print(f"[{r.level.value}] {r.rule}: {r.message}")
```

### Compute statistics

```python
stats = pyap.compute_stats(dataset)
print(stats.summary())

# Individual stats
print(f"Samples: {stats.n_samples}")
print(f"Features: {stats.n_features}")
print(f"Detection rate: {stats.detection_rate:.1%}")
```

### Fetch from PRIDE Archive

```python
client = pyap.PrideClient()
project = client.get_project("PAD000001")
files = client.list_files("PAD000001")
urls = client.get_download_urls("PAD000001")
```

## Supported File Formats

| Format | Platform | Reader |
|--------|----------|--------|
| `.npx.csv` | Olink Explore / Target | `pyap.readers.read_olink_csv()` |
| `.parquet` | Olink Explore HT | `pyap.readers.read_olink_parquet()` |
| `.xlsx` | Olink | `pyap.readers.read_olink_xlsx()` |
| `.adat` | SomaScan | `pyap.readers.read_somascan_adat()` |
| `.csv` (SomaScan) | SomaScan | `pyap.readers.read_somascan_csv()` |

## Validation Rules

Validators check datasets against the [PRIDE Affinity Proteomics Guidelines](https://github.com/PRIDE-Archive/pride-resources/blob/main/guidelines/pride-affinity-guidelines.md):

- **Schema validation**: required columns, data types
- **QC consistency**: SampleQC values match NPX/RFU data
- **Value ranges**: NPX log2-scale bounds, RFU positivity
- **Data completeness**: non-empty expression matrix

## Data Model

All readers produce an `AffinityDataset`:

```python
@dataclass
class AffinityDataset:
    platform: Platform          # OLINK_EXPLORE, OLINK_TARGET, SOMASCAN, etc.
    samples: pd.DataFrame       # Sample metadata
    features: pd.DataFrame      # Protein/aptamer annotations
    expression: pd.DataFrame    # Quantification matrix (NPX or RFU)
    metadata: dict              # Platform-specific extras
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/pyap/
```

## License

Apache License 2.0
