# Importing SwingVision data safely

## Current evidence boundary

SwingVision publicly advertises raw-data export for Pro users and documents statistics such as shot speed, consistency, first-serve/groundstroke percentages, winners/errors and break-point information.

The public product documentation consulted for this build does **not** define one stable machine-readable schema for every export type/version. Third-party tools report spreadsheet/CSV exports, but this repository deliberately avoids hard-coding their schema as if it were an official contract.

## Recommended first real-data step

When I add one real export of my own:

```bash
PYTHONPATH=src python -m tennis_ai.inspect_export data/private/my_export.xlsx --out reports/private_schema.json
```

The inspector prints sheet and column names only. It does not print row values.

Then build a project-specific converter into the canonical session schema and add tests using a small anonymised fixture.

## Canonical fields

Minimum current fields:

- `session_id`
- `date`
- `session_type`
- `result`
- `serve_points`
- `first_serves_in`
- `double_faults`
- `winners`
- `unforced_errors`
- `forehand_in`, `forehand_attempted`
- `backhand_in`, `backhand_attempted`
- `avg_serve_speed_kmh`
- `rallies_over_5`, `rallies_total`

Optional useful fields include opponent, opponent rating/level, point counts, break-point counts, shot speeds and duration.

If an export lacks one of the required fields, do **not** invent it. Either:

1. revise the canonical schema for the actual available data; or
2. create the missing metric only when it can be derived exactly from more granular exported information.

## Privacy

Real exports should remain under `data/private/`, which is git-ignored.

Before committing any anonymised fixture, remove:

- names/usernames;
- exact locations;
- video URLs;
- account IDs;
- free-text notes containing personal information.
