import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from requisitos import exigir_python_312

RAIZ = Path(__file__).resolve().parent.parent


def test_python_312_o_mayor_pasa() -> None:
    exigir_python_312((3, 12, 0))
    exigir_python_312((3, 13, 1))
    exigir_python_312()


def test_python_311_se_rechaza_con_el_mensaje() -> None:
    with pytest.raises(SystemExit) as error:
        exigir_python_312((3, 11, 9))
    assert str(error.value) == (
        "Python 3.12+ requerido: agente.py usa el alias `type` (PEP 695). Versión actual: 3.11. "
        "Creá el venv con `py -3.12 -m venv .venv` (Windows) o `python3.12 -m venv .venv` (Linux/macOS)."
    )


@pytest.mark.parametrize(
    "nombre", ["main.py", "historial.py", "experimento_recursion.py", "validacion.py", "tests/conftest.py"]
)
def test_el_chequeo_corre_antes_de_importar_agente(nombre: str) -> None:
    lineas = (RAIZ / nombre).read_text(encoding="utf-8").splitlines()
    guardia = lineas.index("exigir_python_312()")
    imports_de_agente = [i for i, linea in enumerate(lineas) if re.match(r"(from|import) agente\b", linea)]
    assert imports_de_agente and guardia < imports_de_agente[0]


def test_requisitos_parsea_con_sintaxis_de_python_3_8() -> None:
    ast.parse((RAIZ / "requisitos.py").read_text(encoding="utf-8"), feature_version=(3, 8))
    with pytest.raises(SyntaxError):
        ast.parse((RAIZ / "agente.py").read_text(encoding="utf-8"), feature_version=(3, 11))


def test_pyproject_y_python_version_piden_312() -> None:
    assert 'requires-python = ">=3.12"' in (RAIZ / "pyproject.toml").read_text(encoding="utf-8")
    assert (RAIZ / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert sys.version_info >= (3, 12)
    assert subprocess.run([sys.executable, "-c", "import requisitos"], cwd=RAIZ).returncode == 0
