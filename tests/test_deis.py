from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pytest

from urgencias_core.data.deis import (
    DEIS_DATA_START_YEAR,
    DEMO_HOSPITALS,
    HOSPITAL_CODE_ALIASES,
    _canonicalize,
    _expand_codes,
    download_year,
    fetch_demo_hospitals,
    year_url,
)

SNAPSHOT = Path(__file__).parent / "fixtures" / "deis_demo_snapshot.parquet"


def test_constants_are_sane() -> None:
    assert DEIS_DATA_START_YEAR == 2008
    assert set(DEMO_HOSPITALS.keys()) == {"hospital_base_puerto_montt", "hospital_frutillar"}
    assert DEMO_HOSPITALS["hospital_base_puerto_montt"] == "124105"
    assert DEMO_HOSPITALS["hospital_frutillar"] == "124115"
    assert "24-105" in HOSPITAL_CODE_ALIASES["124105"]
    assert "24-115" in HOSPITAL_CODE_ALIASES["124115"]


def test_expand_codes_includes_aliases() -> None:
    codes = _expand_codes(DEMO_HOSPITALS)
    assert {"124105", "24-105", "124115", "24-115"}.issubset(codes)


def test_year_url_pattern() -> None:
    assert (
        year_url(2025)
        == "https://repositoriodeis.minsal.cl/SistemaAtencionesUrgencia/AtencionesUrgencia2025.zip"
    )


def test_canonicalize_matches_raw_schema() -> None:
    raw = pd.DataFrame(
        {
            "IdEstablecimiento": ["24-105", "24-115"],
            "NEstablecimiento": ["Hospital de Puerto Montt", "Hospital de Frutillar"],
            "IdCausa": ["1", "1"],
            "GlosaCausa": ["SECCIÓN 1. TOTAL ATENCIONES DE URGENCIA", "SECCIÓN 1. TOTAL ATENCIONES DE URGENCIA"],
            "Total": ["800", "80"],
            "fecha": ["01/03/2025", "01/03/2025"],
        }
    )
    out = _canonicalize(raw, 2025)
    assert len(out) == 2
    assert set(out["facility_code"]) == {"24-105", "24-115"}
    assert out["date"].dt.year.tolist() == [2025, 2025]
    assert out["count"].tolist() == [800, 80]
    assert out["cause_group"].iloc[0].startswith("SECCI")


def test_canonicalize_drops_unparseable_dates() -> None:
    raw = pd.DataFrame(
        {
            "IdEstablecimiento": ["24-105"],
            "fecha": ["not-a-date"],
            "Total": ["50"],
        }
    )
    out = _canonicalize(raw, 2025)
    assert len(out) == 0


def test_canonicalize_raises_on_schema_drift() -> None:
    from urgencias_core.data.deis import SchemaDriftError

    raw = pd.DataFrame({"Foo": [1], "Bar": [2]})
    with pytest.raises(SchemaDriftError):
        _canonicalize(raw, 2099)


def test_fetch_demo_hospitals_offline_fallback(tmp_path: Path) -> None:
    # Force the live fetch to find nothing by using a client that always 404s.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    df = fetch_demo_hospitals(
        start_year=2025,
        end_year=2025,
        cache_dir=tmp_path / "deis_cache",
        client=client,
        snapshot_path=SNAPSHOT,
    )
    client.close()
    assert len(df) > 0
    assert {"24-105", "24-115"}.issubset(set(df["facility_code"].astype(str)))


def test_snapshot_shape_and_coverage() -> None:
    assert SNAPSHOT.exists()
    df = pd.read_parquet(SNAPSHOT)
    assert len(df) > 100_000
    years = set(int(y) for y in df["year"].unique())
    assert {2022, 2023, 2024, 2025}.issubset(years)
    assert set(df["facility_code"].astype(str)) == {"24-105", "24-115"}
    assert df["date"].notna().all()


def test_download_year_cache_hit_no_refetch(tmp_path: Path) -> None:
    from urgencias_core.data.deis import DEIS_ZIP_PATTERN

    cache = tmp_path / "deis_cache"
    cache.mkdir()
    # Pre-populate a fake cache file for a closed year.
    fake = cache / DEIS_ZIP_PATTERN.format(year=2022)
    # Build a minimal valid zip with an empty CSV so existence check suffices.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("AtencionesUrgencia2022.csv", "IdEstablecimiento;fecha;Total\n")
    fake.write_bytes(buf.getvalue())

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_year(2022, cache_dir=cache, client=client, current_year=2026)
    client.close()
    assert result.status == "cached"
    assert calls == []
