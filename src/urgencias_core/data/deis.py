"""DEIS Atenciones de Urgencia fetcher, parser, and hospital filter.

Source
------
DEIS — Departamento de Estadísticas e Información de Salud, Ministerio de Salud
de Chile. Open data, no authentication required.

Annual ZIP archives are published at::

    https://repositoriodeis.minsal.cl/SistemaAtencionesUrgencia/AtencionesUrgencia{YEAR}.zip

Coverage 2008 to the present. The current-year file is partial and is updated
weekly during the winter respiratory campaign (March–September) and roughly
monthly outside it. Historical closed-year files are static.

Each ZIP contains a single CSV with aggregated daily counts per establishment
per cause group per age group. The exact CSV filename and column layout has
drifted year over year; see ``SCHEMAS`` and ``_canonicalize`` for the per-year
mapping to a canonical schema.

Attribution
-----------
If you use this code against real DEIS data, credit DEIS MINSAL
(``https://deis.minsal.cl/``) under Chile's open data framework.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

DEIS_BASE_URL = "https://repositoriodeis.minsal.cl/SistemaAtencionesUrgencia"
DEIS_ZIP_PATTERN = "AtencionesUrgencia{year}.zip"
DEIS_DATA_START_YEAR = 2008
DEFAULT_CACHE_DIR = Path("data/external/deis_cache")
CURRENT_YEAR_MAX_CACHE_DAYS = 7

# Demo facility codes. Verified against the DEIS MINSAL establecimientos
# xlsx (2019-04-01 snapshot from ssmaule.gob.cl) and cross-checked via the
# public DEIS establishment list. SS Reloncaví.
DEMO_HOSPITALS: dict[str, str] = {
    "hospital_base_puerto_montt": "124105",  # official DEIS name: Hospital de Puerto Montt
    "hospital_frutillar": "124115",
}

_CANONICAL_COLUMNS = (
    "year", "date", "facility_code", "facility_name",
    "cause_id", "cause_group", "age_group", "count",
)

# Legacy "código antiguo" aliases. Older DEIS files use "XX-YYY" form, newer
# files use the 6-digit "nuevo" code. Match either at filter time.
HOSPITAL_CODE_ALIASES: dict[str, list[str]] = {
    "124105": ["24-105", "105"],
    "124115": ["24-115", "115"],
}


@dataclass(frozen=True)
class YearFetchResult:
    year: int
    path: Path | None
    status: str  # "downloaded", "cached", "not_available", "error"
    detail: str = ""


def year_url(year: int) -> str:
    return f"{DEIS_BASE_URL}/{DEIS_ZIP_PATTERN.format(year=year)}"


def _retry_get_stream(
    url: str,
    dest: Path,
    client: httpx.Client,
    retries: int = 3,
) -> tuple[int, str]:
    """Download ``url`` to ``dest`` with streaming + exponential backoff on 5xx/timeouts.

    Returns (status_code, detail_string).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    delays = [2, 4, 8]
    for attempt in range(retries):
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
                if resp.status_code == 200:
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=1 << 15):
                            f.write(chunk)
                    tmp.replace(dest)
                    return 200, "ok"
                if resp.status_code == 404:
                    return 404, "not_available"
                if 500 <= resp.status_code < 600 and attempt < retries - 1:
                    time.sleep(delays[attempt])
                    continue
                return resp.status_code, f"http_{resp.status_code}"
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt < retries - 1:
                time.sleep(delays[attempt])
                continue
            return 0, f"transport_error:{type(e).__name__}"
    return 0, "exhausted"


def _cache_is_fresh(path: Path, year: int, current_year: int) -> bool:
    if not path.exists():
        return False
    if year < current_year:
        return True
    age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400.0
    return age_days <= CURRENT_YEAR_MAX_CACHE_DAYS


def download_year(
    year: int,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client: httpx.Client | None = None,
    *,
    current_year: int | None = None,
) -> YearFetchResult:
    """Download a single year's ZIP to the cache directory.

    Returns a ``YearFetchResult`` whose ``status`` is one of:
    - ``"cached"``: file was already fresh, no download.
    - ``"downloaded"``: file was downloaded successfully.
    - ``"not_available"``: server returned 404 (year not yet published or withdrawn).
    - ``"error"``: other failure; ``detail`` has more info.
    """
    cache_dir = Path(cache_dir)
    dest = cache_dir / DEIS_ZIP_PATTERN.format(year=year)
    current = current_year or datetime.now().year

    if _cache_is_fresh(dest, year, current):
        return YearFetchResult(year=year, path=dest, status="cached")

    url = year_url(year)
    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        code, detail = _retry_get_stream(url, dest, client)
    finally:
        if owns_client:
            client.close()

    if code == 200:
        return YearFetchResult(year=year, path=dest, status="downloaded")
    if code == 404:
        return YearFetchResult(year=year, path=None, status="not_available", detail=detail)
    return YearFetchResult(year=year, path=None, status="error", detail=detail)


_FACILITY_COL_CANDIDATES = (
    "IdEstablecimiento",
    "idestablecimiento",
    "Cod_Estab",
    "cod_establecimiento",
    "CodigoEstablecimiento",
    "Establecimiento",
    "cod_esta",
    "Cod_Esta",
)


def _detect_facility_column(cols: list[str]) -> str | None:
    lowered = {c.lower(): c for c in cols}
    for cand in _FACILITY_COL_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _read_csv_chunks(path: Path, *, chunksize: int = 200_000):
    """Yield (encoding, chunk) pairs reading the first CSV member in ``path``."""
    with zipfile.ZipFile(path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV found inside {path.name}. Members: {zf.namelist()}")
        member = csv_members[0]
        # DEIS historically uses latin-1 with semicolon separators. Probe once.
        encodings = ("latin-1", "utf-8-sig", "utf-8")
        for encoding in encodings:
            try:
                with zf.open(member) as f:
                    reader = pd.read_csv(
                        f,
                        sep=";",
                        encoding=encoding,
                        dtype=str,
                        chunksize=chunksize,
                        low_memory=False,
                    )
                    for chunk in reader:
                        yield member, chunk
                    return
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode CSV in {path.name}")


def _canonicalize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Normalize a raw DEIS year frame to the canonical column set.

    Canonical columns:
        year              int
        date              datetime64 (daily grain)
        facility_code     str  (code as published; may be antiguo or nuevo)
        facility_name     str
        cause_id          str  (empty when source lacks it)
        cause_group       str
        age_group         str  (empty for wide-schema years with per-age columns)
        count             int  (total count; age-breakdown columns are dropped)
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            key = cand.lower().strip()
            if key in cols:
                return cols[key]
        return None

    facility_code_col = pick(
        "establecimiento", "idestablecimiento", "cod_establecimiento",
        "codigoestablecimiento", "codigo_establecimiento", "cod_estab",
        "cod_esta", "establecimiento_cod",
    )
    facility_name_col = pick(
        "nestablecimiento", "nombreestablecimiento", "nombre_establecimiento",
        "estab_nombre", "glosaestablecimiento", "establecimiento_glosa",
    )
    date_col = pick("fecha", "fechaatencion", "fecha_atencion", "fecha_registro")
    cause_col = pick(
        "glosacausa", "causa", "nombrecausa", "causa_glosa", "grupo_causa",
        "grupo_de_causa", "descripcioncausa", "nombre_causa",
    )
    cause_id_col = pick("idcausa", "cod_causa", "codigocausa", "causa_id")
    age_col = pick(
        "grupo_edad", "grupoedad", "grupo_de_edad", "tramo_edad", "tramoedad",
        "edad_tramo", "edad", "causaedad",
    )
    count_col = pick(
        "total", "numtotal", "numero_total", "numeroatenciones", "atenciones",
        "numero_atenciones", "cantidad", "conteo", "n",
    )

    required = {"facility_code": facility_code_col, "date": date_col, "count": count_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise SchemaDriftError(
            year=year,
            message=f"Could not locate columns {missing} in year {year}. "
            f"Available columns: {sorted(df.columns)}",
        )

    df = df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["year"] = pd.Series([year] * len(df), dtype="int16").values
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).values
    out["facility_code"] = df[facility_code_col].astype(str).str.strip().values
    out["facility_name"] = (
        df[facility_name_col].astype(str).str.strip().values if facility_name_col else ""
    )
    out["cause_id"] = (
        df[cause_id_col].astype(str).str.strip().values if cause_id_col else ""
    )
    out["cause_group"] = df[cause_col].astype(str).str.strip().values if cause_col else ""
    out["age_group"] = df[age_col].astype(str).str.strip().values if age_col else ""
    out["count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype("int64").values

    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


class SchemaDriftError(Exception):
    def __init__(self, year: int, message: str) -> None:
        super().__init__(message)
        self.year = year


def load_year(
    year: int,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client: httpx.Client | None = None,
    *,
    facility_filter: set[str] | None = None,
) -> pd.DataFrame | None:
    """Download (if needed) and parse a single year. Returns None when unavailable.

    When ``facility_filter`` is supplied, only rows whose facility code is in
    the set are retained during chunked parsing. This is essential for large
    years (the 2024 CSV is ~1.6 GB uncompressed; filtering to 2 hospitals cuts
    it to ~60 k rows).
    """
    result = download_year(year, cache_dir=cache_dir, client=client)
    if result.status in ("not_available", "error") or result.path is None:
        if result.status == "not_available":
            logger.warning("DEIS year %d not available (404). Skipping.", year)
        else:
            logger.warning("DEIS year %d failed: %s. Skipping.", year, result.detail)
        return None

    frames: list[pd.DataFrame] = []
    try:
        for _, chunk in _read_csv_chunks(result.path):
            if facility_filter is not None:
                fcol = _detect_facility_column(list(chunk.columns))
                if fcol is None:
                    raise SchemaDriftError(
                        year=year,
                        message=f"No facility column detected. Columns: {sorted(chunk.columns)}",
                    )
                chunk = chunk.loc[
                    chunk[fcol].astype(str).str.strip().isin(facility_filter)
                ]
                if len(chunk) == 0:
                    continue
            frames.append(_canonicalize(chunk, year))
        if not frames:
            return pd.DataFrame(columns=_CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)
    except SchemaDriftError as e:
        logger.warning("Schema drift in year %d: %s", e.year, e)
        return None
    except Exception as e:
        logger.warning("Parse failure in year %d: %s", year, e)
        return None


def fetch(
    start_year: int = DEIS_DATA_START_YEAR,
    end_year: int | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client: httpx.Client | None = None,
    *,
    facility_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Fetch and parse all available years in [start_year, end_year].

    When ``end_year`` is ``None``, defaults to the current calendar year.
    Years that return 404, fail to download, or hit schema drift are logged
    and skipped — the function does not raise on partial failure.

    ``facility_filter`` pushes the hospital filter down into chunked parsing
    to keep memory use low; callers that want all facilities should leave it
    as ``None`` and expect multi-GB in-memory frames.
    """
    current = datetime.now().year
    end = end_year or current
    frames: list[pd.DataFrame] = []

    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        for year in range(start_year, end + 1):
            df = load_year(
                year,
                cache_dir=cache_dir,
                client=client,
                facility_filter=facility_filter,
            )
            if df is not None and len(df) > 0:
                frames.append(df)
    finally:
        if owns_client:
            client.close()

    if not frames:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _expand_codes(codes: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for code in codes.values():
        out.add(code)
        out.update(HOSPITAL_CODE_ALIASES.get(code, []))
    return out


def fetch_demo_hospitals(
    start_year: int = DEIS_DATA_START_YEAR,
    end_year: int | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client: httpx.Client | None = None,
    *,
    snapshot_path: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch the full DEIS dataset and filter to the two demo hospitals.

    ``snapshot_path`` is used as an offline fallback: if the live fetch returns
    no rows (all years 404 or network failure), the snapshot parquet is
    loaded instead. This keeps the demo reproducible in CI and on airplane
    mode.
    """
    codes = _expand_codes(DEMO_HOSPITALS)
    try:
        df = fetch(
            start_year=start_year,
            end_year=end_year,
            cache_dir=cache_dir,
            client=client,
            facility_filter=codes,
        )
    except Exception as e:
        logger.warning("Live fetch failed: %s. Falling back to snapshot.", e)
        df = pd.DataFrame()

    if len(df) == 0 and snapshot_path is not None:
        sp = Path(snapshot_path)
        if sp.exists():
            logger.info("Loading offline DEIS snapshot from %s", sp)
            return pd.read_parquet(sp)

    return df.reset_index(drop=True)


__all__ = [
    "DEIS_BASE_URL",
    "DEIS_DATA_START_YEAR",
    "DEIS_ZIP_PATTERN",
    "DEMO_HOSPITALS",
    "HOSPITAL_CODE_ALIASES",
    "SchemaDriftError",
    "YearFetchResult",
    "download_year",
    "fetch",
    "fetch_demo_hospitals",
    "load_year",
    "year_url",
]
