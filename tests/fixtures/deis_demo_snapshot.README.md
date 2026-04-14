# deis_demo_snapshot.parquet

Frozen offline snapshot of DEIS MINSAL *Atenciones de Urgencia* open data,
filtered to the two demo hospitals in the Los Lagos region:

- `24-105` — Hospital de Puerto Montt (also known as Hospital Base de Puerto
  Montt). High-complexity. Servicio de Salud Reloncaví.
- `24-115` — Hospital de Frutillar. Low-complexity. Servicio de Salud Reloncaví.

## Generation

- Fetched by `urgencias_core.data.deis.fetch_demo_hospitals(start_year=2021,
  end_year=2026)` on **2026-04-13**.
- Source URL pattern: `https://repositoriodeis.minsal.cl/SistemaAtencionesUrgencia/AtencionesUrgencia{YEAR}.zip`
- Encoding: semicolon-separated CSV, latin-1.

## Coverage

| Year | Rows | Note |
|------|------|------|
| 2021 | 27,760 | Complete year |
| 2022 | 28,720 | Complete year |
| 2023 | 28,760 | Complete year |
| 2024 | 28,960 | Complete year |
| 2025 | 28,749 | Complete year |
| 2026 | 8,159 | Partial year, through 2026-04-11 (file updated weekly during the winter campaign) |

Total 151,108 rows, daily grain, one row per (hospital, date, cause group).

## Years not included

- **2017–2019**: DEIS published these as `.xlsx` + `.mdb` bundles inside the
  same annual ZIP. The CSV parser in `deis.py` skips them. Adding
  `.xlsx` / `.mdb` support is tracked in `docs/roadmap.md`.
- **2020**: the ZIP contains a malformed header row (the source export
  mangled the separator line). Excluded by default per the COVID handling
  decision in `docs/decisions.md` — but would be skipped by the parser
  anyway.

## Regeneration

```bash
rm tests/fixtures/deis_demo_snapshot.parquet
uv run python -c "
from urgencias_core.data.deis import fetch_demo_hospitals
from pathlib import Path
df = fetch_demo_hospitals(start_year=2021)
df.to_parquet('tests/fixtures/deis_demo_snapshot.parquet', index=False, compression='zstd')
print(len(df), 'rows')
"
```

## Attribution

DEIS — Departamento de Estadísticas e Información de Salud, Ministerio de
Salud de Chile. `https://deis.minsal.cl/`. Open data under Chile's open data
framework. This repository redistributes a filtered subset for teaching and
reproducibility; for any research or operational use of the full dataset,
fetch directly from the source.
