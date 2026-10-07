from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

import agente
from agente import AppAgente, abrir_agente
from tests.modelo_local import ModeloPedidosLocal


@pytest.fixture(autouse=True)
def modelo_local(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModeloPedidosLocal]:
    """El grafo usa el doble en vez del LLM real y el backoff no espera."""
    doble = ModeloPedidosLocal()
    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: doble)
    monkeypatch.setattr(agente, "ESPERA_INICIAL_S", 0)
    yield doble


@pytest.fixture
def ruta_db(tmp_path: Path) -> str:
    return str(tmp_path / "checkpoints.sqlite")


@pytest.fixture
async def app(ruta_db: str) -> AsyncIterator[AppAgente]:
    async with abrir_agente(ruta_db) as compilado:
        yield compilado
