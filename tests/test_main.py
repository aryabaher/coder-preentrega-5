import json
import subprocess
import sys
from pathlib import Path

import pytest

import historial
import main

RAIZ = Path(__file__).resolve().parent.parent


async def test_demo_completa_con_el_doble(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ruta_db = str(tmp_path / "checkpoints.sqlite")
    await main.main(ruta_db, tmp_path / "traza.json", tmp_path / "traza.log")
    salida = capsys.readouterr().out
    assert "--- conversacion-demo-1: 16 mensajes en" in salida
    assert "================================ Human Message =================================" in salida

    turnos = json.loads((tmp_path / "traza.json").read_text(encoding="utf-8"))["turnos"]
    assert list(turnos) == [nombre for nombre, _, _ in main.ESCENARIOS]
    assert len(turnos["turno_1_multi_paso"]["herramientas_invocadas"]) == 2
    assert turnos["turno_2_memoria"]["herramientas_invocadas"] == ["buscar_ultimo_pedido(cliente_id=102)"]
    assert turnos["turno_3_memoria_otro_cliente"]["herramientas_invocadas"] == [
        "buscar_cliente_por_nombre(nombre='Juan Pérez')",
        "buscar_ultimo_pedido(cliente_id=205)",
    ]
    assert len(turnos["prueba_reintento_nombre_incompleto"]["herramientas_invocadas"]) == 3
    assert "confirmar el nombre completo" in turnos["prueba_error_y_aclaracion"]["respuesta"]
    assert turnos["prueba_hilo_aislado"]["herramientas_invocadas"] == []


async def test_correr_dos_veces_no_duplica_el_historial(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ruta_db = str(tmp_path / "checkpoints.sqlite")
    for _ in range(2):
        await main.main(ruta_db, tmp_path / "traza.json", tmp_path / "traza.log")
    assert capsys.readouterr().out.count("--- conversacion-demo-1: 16 mensajes en") == 2


async def test_historial_recupera_el_thread_en_otra_apertura(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ruta_db = str(tmp_path / "checkpoints.sqlite")
    await main.main(ruta_db, tmp_path / "traza.json", tmp_path / "traza.log")
    capsys.readouterr()
    assert await historial.mostrar_historial("conversacion-demo-1", "¿Y el último?", ruta_db) == 16
    salida = capsys.readouterr().out
    assert "--- conversacion-demo-1: 16 mensajes recuperados de" in salida
    assert "Herramientas: ['buscar_ultimo_pedido(cliente_id=205)']" in salida
    assert await historial.mostrar_historial("no-existe", ruta_db=ruta_db) == 0


def test_main_sin_clave_sale_con_401(tmp_path: Path) -> None:
    entorno = {"PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"}
    proceso = subprocess.run(
        [sys.executable, str(RAIZ / "main.py")],
        cwd=tmp_path,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proceso.returncode == 1
    assert proceso.stdout.strip().endswith(
        "Error controlado: 401/key: falta OPENAI_API_KEY en las variables de entorno. "
        "Copiá .env.example a .env y cargá la clave. No se reintenta."
    )


def test_sin_claves_ni_dobles_en_el_codigo_entregado() -> None:
    for nombre in ("main.py", "agente.py", "modelo.py", "herramientas.py", "historial.py"):
        fuente = (RAIZ / nombre).read_text(encoding="utf-8")
        assert "sk-" not in fuente
        assert "ModeloPedidosLocal" not in fuente
    fuente_agente = (RAIZ / "agente.py").read_text(encoding="utf-8")
    assert fuente_agente.count("StateGraph(MessagesState)") == 1
    assert "StateGraph" not in (RAIZ / "main.py").read_text(encoding="utf-8")
    gitignore = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore and "*.sqlite" in gitignore
