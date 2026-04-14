# Roadmap

Items diferidos conscientemente durante el bootstrap v1. No son bugs;
son decisiones de no implementar algo ahora por costo/valor o por
dependencias externas no resueltas.

## Soporte para archivos DEIS 2017–2019 (xlsx + mdb)

El fetcher DEIS actual (`urgencias_core.data.deis`) solo parsea CSV.
Las publicaciones DEIS 2017–2019 vinieron como archivos `.xlsx` +
`.mdb` dentro del ZIP anual. El parser los salta con una advertencia.

Agregar soporte extendería la cobertura del snapshot demo hacia atrás
a 2017, pero dado que la política COVID ya excluye 2020–2021, el valor
incremental es limitado a menos que un usuario quiera entrenar sobre
el período pre-pandémico explícitamente.

- `.xlsx`: `openpyxl` (ya disponible si se agrega a deps), una sheet
  por facility o una sheet única tipo flat; revisar al implementar.
- `.mdb`: `mdbtools` (dep de sistema macOS/Linux), o `pandas-access`
  (wrapper de mdbtools). Complica el bootstrap en macOS.

## Parser del archivo DEIS 2020

El CSV de 2020 tiene una fila de cabecera corrupta en la fuente (mezcla
nombres de columna con valores de datos). El parser actual levanta
`SchemaDriftError` y salta el año. Si alguien necesita 2020
explícitamente, se puede escribir un parser específico que ignore la
primera fila y use un mapeo de columnas fijo para ese año. Valor bajo
porque 2020 está excluido por la política COVID por defecto.

## Backtesting rolling en lugar de holdout simple

El harness actual hace un holdout simple (las últimas N observaciones
son test, todo lo anterior es train). Un backtest rolling (sliding
window, refit cada K pasos) da métricas estadísticamente más robustas y
detecta model drift en el tiempo. No es crítico para v1.

## Integración EMR para separar workup de boarding

Ver `docs/decisions.md` sección "LOS, boarding, y qué nos dicen (y no
nos dicen) los datos". Ingerir el timestamp de decisión de admisión
desde EMR permitiría distinguir tiempo clínico activo del tiempo
de espera por cama. Requiere o una nueva columna en el export del EMR o
un proxy por timestamps de entrada de órdenes. Sprint 3 candidato en
el plan de `eunosia-forecast`.

## Console scripts / entry points

`pyproject.toml` no define console scripts. El demo y el servidor
se invocan via `uv run python scripts/demo_*.py` y
`uv run uvicorn urgencias_core.server.app:app`. Es intencional: este
código no pretende ser un CLI instalable.

## Console logging unificado

Varios módulos usan `print()` (demos) y otros usan `logging.getLogger(__name__)`
(deis, simulation). Estandarizar sobre logging con un handler por
defecto en los demos haría los usos más configurables. No es
crítico.
