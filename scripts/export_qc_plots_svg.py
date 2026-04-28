#!/usr/bin/env python3
"""Export NPX Distribution and Expression Heatmap figures as SVG (publication-sized).

Uses the same computation and rendering as the QC HTML report, then writes static
vector output via Kaleido. Requires: pip install pyprideap[plots] (includes kaleido).

Examples:
  python scripts/export_qc_plots_svg.py path/to/data.npx.csv -o figures/
  python scripts/export_qc_plots_svg.py -a PAD000033 -o figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Single canvas size for all SVG exports (matches NPX distribution figure).
_EXPORT_WIDTH_PX = 2000
_EXPORT_HEIGHT_PX = 1100


def _download_pad_files(accession: str, dest_dir: Path) -> list[Path]:
    from pyprideap.api.pride import PrideClient

    client = PrideClient()
    urls = client.get_download_urls(accession)
    data_extensions = {".csv", ".parquet", ".xlsx", ".adat"}
    data_urls = {
        name: url
        for name, url in urls.items()
        if any(name.lower().endswith(ext) for ext in data_extensions) and "checksum" not in name.lower()
    }
    if not data_urls:
        logger.error("No supported data files for %s. Available: %s", accession, list(urls.keys()))
        sys.exit(1)

    out: list[Path] = []
    for name, url in data_urls.items():
        dest = dest_dir / Path(name).name
        if url.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            url = url.replace("ftp://ftp.pride.ebi.ac.uk/", "https://ftp.pride.ebi.ac.uk/", 1)
        logger.info("Downloading %s ...", name)
        urlretrieve(url, dest)
        out.append(dest)
    return out


def _layout_distribution_svg(fig, n_samples: int) -> None:
    """Large canvas and typography for vector export."""
    bottom = 160 if n_samples > 10 else 100
    fig.update_layout(
        template="plotly_white",
        width=_EXPORT_WIDTH_PX,
        height=_EXPORT_HEIGHT_PX,
        font=dict(size=18, family="Arial, sans-serif"),
        title=dict(font=dict(size=22)),
        xaxis=dict(title_font=dict(size=19), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=19), tickfont=dict(size=14)),
        legend=dict(font=dict(size=13)),
        margin=dict(l=80, r=40, t=100, b=bottom),
    )


def _layout_heatmap_svg(fig, n_rows: int, n_cols: int) -> None:
    tick_x = n_cols <= 60
    tick_y = n_rows <= 80
    fig.update_layout(
        template="plotly_white",
        width=_EXPORT_WIDTH_PX,
        height=_EXPORT_HEIGHT_PX,
        font=dict(size=18, family="Arial, sans-serif"),
        title=dict(font=dict(size=22)),
        xaxis=dict(
            title_font=dict(size=19),
            tickfont=dict(size=9 if n_cols > 80 else 11),
            showticklabels=tick_x,
        ),
        yaxis=dict(
            title_font=dict(size=19),
            tickfont=dict(size=9 if n_rows > 100 else 11),
            showticklabels=tick_y,
        ),
        margin=dict(l=120, r=100, t=100, b=200 if tick_x else 120),
    )
    fig.update_traces(
        colorbar=dict(title_font=dict(size=17), tickfont=dict(size=14)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NPX Distribution and Expression Heatmap as SVG.")
    parser.add_argument("input_file", nargs="?", help="Local NPX / ADAT / parquet file")
    parser.add_argument("-a", "--accession", help="PAD accession (download from PRIDE)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory for .svg files")
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=200,
        help="Heatmap: most variable proteins to include (default 200, same as QC report)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Heatmap: max samples (default: effectively all rows)",
    )
    parser.add_argument("-p", "--platform", choices=["olink", "somascan"], default=None)
    args = parser.parse_args()

    if (args.input_file is None) == (args.accession is None):
        parser.error("Provide exactly one of: input_file or --accession")

    try:
        import plotly  # noqa: F401
    except ImportError as e:
        raise SystemExit("Plotly is required. Install with: pip install pyprideap[plots]") from e

    import pyprideap as pp
    from pyprideap.viz.qc import render as R
    from pyprideap.viz.qc.compute import compute_distribution, compute_heatmap

    if args.accession:
        accession = args.accession.upper()
        with tempfile.TemporaryDirectory(prefix="pyprideap_svg_") as tmp:
            files = _download_pad_files(accession, Path(tmp))
            data_path = files[0]
            ds = pp.read(data_path, platform=args.platform)
    else:
        data_path = Path(args.input_file)
        if not data_path.exists():
            raise SystemExit(f"File not found: {data_path}")
        ds = pp.read(data_path, platform=args.platform)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = data_path.stem
    if stem.endswith(".npx") or stem.endswith(".ct"):
        stem = Path(stem).stem

    dist_data = compute_distribution(ds)
    fig_d = R.render_distribution(dist_data)
    _layout_distribution_svg(fig_d, n_samples=len(dist_data.sample_ids))

    heat_data = compute_heatmap(ds, max_proteins=args.max_proteins, max_samples=args.max_samples)
    if heat_data is None:
        raise SystemExit("Could not compute expression heatmap (dataset too small or empty).")

    fig_h = R.render_heatmap(heat_data)
    z = fig_h.data[0].z
    n_rows = len(z)
    n_cols = len(z[0]) if n_rows else 0
    _layout_heatmap_svg(fig_h, n_rows=n_rows, n_cols=n_cols)

    dist_path = out_dir / f"{stem}_npx_distribution.svg"
    heat_path = out_dir / f"{stem}_expression_heatmap.svg"

    try:
        fig_d.write_image(str(dist_path), format="svg")
        fig_h.write_image(str(heat_path), format="svg")
    except ValueError as e:
        if "kaleido" in str(e).lower() or "orca" in str(e).lower():
            raise SystemExit(
                "Static image export needs Kaleido. Install with:\n  pip install kaleido\n"
                "or  pip install \"pyprideap[plots]\""
            ) from e
        raise

    logger.info("Wrote %s", dist_path.resolve())
    logger.info("Wrote %s", heat_path.resolve())


if __name__ == "__main__":
    main()
