"""Validate a scenario JSON file against the Pydantic schema."""

import json
import sys
from pathlib import Path

import click


@click.command()
@click.argument("scenario_file", type=click.Path(exists=True))
def validate(scenario_file: str) -> None:
    """Validate a scenario JSON file."""
    from longhorizon_bench.schema import Scenario

    path = Path(scenario_file)
    click.echo(f"Validating {path.name}...")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    try:
        scenario = Scenario.model_validate(raw)
    except Exception as e:
        click.echo(f"FAIL: {e}", err=True)
        sys.exit(1)

    action_count = len(scenario.action_events)
    checkpoint_count = len(scenario.checkpoint_events)

    warnings: list[str] = []
    if scenario.total_events != action_count:
        warnings.append(
            f"total_events={scenario.total_events} but found {action_count} action events"
        )
    if scenario.total_checkpoints != checkpoint_count:
        warnings.append(
            f"total_checkpoints={scenario.total_checkpoints} but found {checkpoint_count} checkpoints"
        )

    ids = [e.event_id for e in scenario.events]
    if len(ids) != len(set(ids)):
        warnings.append("Duplicate event IDs found")

    id_set = set(ids)
    for event in scenario.action_events:
        for dep in event.depends_on:
            if dep not in id_set:
                warnings.append(f"Event {event.event_id} depends on unknown {dep}")

    if warnings:
        click.echo(f"PASS with {len(warnings)} warning(s):")
        for w in warnings:
            click.echo(f"  - {w}")
    else:
        click.echo(f"PASS: {scenario.scenario_id} ({action_count} actions, {checkpoint_count} checkpoints)")


if __name__ == "__main__":
    validate()
