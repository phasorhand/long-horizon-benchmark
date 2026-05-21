"""CLI entry point for the data construction pipeline."""

import click


@click.group()
def cli() -> None:
    """LongHorizon-Bench data construction pipeline."""
    pass


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--subsets", default="petrochemical,mining,fire_safety_food_safety", help="Comma-separated IndustryCorpus2 subsets")
def download(data_dir: str, subsets: str) -> None:
    """Stage 1: Download and filter corpora."""
    click.echo(f"Downloading to {data_dir}/raw_corpus/ ...")
    from longhorizon_bench.pipeline.downloader import run_download
    run_download(data_dir=data_dir, subsets=subsets.split(","))
    click.echo("Download complete.")


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--n-clusters", default=12, help="Number of topic clusters")
def extract(data_dir: str, n_clusters: int) -> None:
    """Stage 2: Extract atomic events and compose chains."""
    click.echo(f"Extracting skeletons from {data_dir}/raw_corpus/ ...")
    from longhorizon_bench.pipeline.chain_composer import run_extract
    run_extract(data_dir=data_dir, n_clusters=n_clusters)
    click.echo("Extraction complete.")


@cli.command()
@click.option("--data-dir", default="data", help="Base data directory")
def generate(data_dir: str) -> None:
    """Stage 3: Generate full scenarios from skeletons."""
    click.echo(f"Generating scenarios from {data_dir}/skeletons/ ...")
    from longhorizon_bench.pipeline.assembler import run_generate
    run_generate(data_dir=data_dir)
    click.echo("Generation complete.")


@cli.command("validate")
@click.option("--data-dir", default="data", help="Base data directory")
def validate_cmd(data_dir: str) -> None:
    """Stage 4: Validate generated scenarios."""
    click.echo(f"Validating scenarios in {data_dir}/scenarios/ ...")
    from longhorizon_bench.pipeline.simulation_validator import run_validate
    run_validate(data_dir=data_dir)
    click.echo("Validation complete.")


@cli.command("run-all")
@click.option("--data-dir", default="data", help="Base data directory")
@click.option("--subsets", default="petrochemical,mining,fire_safety_food_safety")
@click.option("--n-clusters", default=12)
def run_all(data_dir: str, subsets: str, n_clusters: int) -> None:
    """Run the full pipeline: download → extract → generate → validate."""
    from longhorizon_bench.pipeline.downloader import run_download
    from longhorizon_bench.pipeline.chain_composer import run_extract
    from longhorizon_bench.pipeline.assembler import run_generate
    from longhorizon_bench.pipeline.simulation_validator import run_validate

    click.echo("=== Stage 1: Download ===")
    run_download(data_dir=data_dir, subsets=subsets.split(","))
    click.echo("=== Stage 2: Extract ===")
    run_extract(data_dir=data_dir, n_clusters=n_clusters)
    click.echo("=== Stage 3: Generate ===")
    run_generate(data_dir=data_dir)
    click.echo("=== Stage 4: Validate ===")
    run_validate(data_dir=data_dir)
    click.echo("=== Pipeline complete ===")
