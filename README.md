# urgencias-core

Código de referencia para análisis, simulación y forecasting de servicios de
urgencia en Chile. Fundación abierta de Eunosia.

> Bootstrap en progreso. Este README se completará en la fase 6 con la tabla
> de contenidos, el quickstart y la versión en inglés.

## Quickstart (borrador)

```bash
git clone https://github.com/nicoveraz/urgencias-core
cd urgencias-core
uv sync
uv run python scripts/demo.py
uv run uvicorn urgencias_core.server.app:app
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
