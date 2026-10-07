import json

import pytest
from pydantic import ValidationError

from herramientas import (
    PEDIDOS_DB,
    buscar_cliente_por_nombre,
    buscar_pedidos,
    buscar_ultimo_pedido,
    herramientas,
)


def test_cliente_por_nombre_completo_devuelve_el_id() -> None:
    salida = json.loads(buscar_cliente_por_nombre.invoke({"nombre": "Ana García"}))
    assert salida == {"nombre": "Ana García", "cliente_id": 102}


def test_cliente_sin_tildes_ni_mayusculas() -> None:
    assert json.loads(buscar_cliente_por_nombre.invoke({"nombre": "  juan   PEREZ "}))["cliente_id"] == 205


def test_nombre_incompleto_trae_la_coincidencia_para_reintentar() -> None:
    salida = buscar_cliente_por_nombre.invoke({"nombre": "García"})
    assert salida.startswith("ERROR: 'García' es un nombre incompleto. Coincide con: Ana García.")
    assert "Volvé a llamar a buscar_cliente_por_nombre con el nombre completo" in salida


def test_cliente_inexistente_pide_confirmar() -> None:
    salida = buscar_cliente_por_nombre.invoke({"nombre": "Roberto Sánchez"})
    assert salida.startswith("ERROR: no se encontró ningún cliente con el nombre 'Roberto Sánchez'.")
    assert "No inventes datos" in salida


def test_nombre_de_solo_espacios_pide_el_nombre() -> None:
    salida = buscar_cliente_por_nombre.invoke({"nombre": "   "})
    assert salida == "ERROR: el nombre está vacío. Pedile al usuario el nombre y apellido del cliente."


def test_buscar_pedidos_devuelve_cantidad_y_total() -> None:
    assert json.loads(buscar_pedidos.invoke({"cliente_id": 102})) == {
        "cliente_id": 102,
        "pedidos": 3,
        "total": 14500,
    }


def test_buscar_pedidos_id_inexistente() -> None:
    salida = buscar_pedidos.invoke({"cliente_id": 999})
    assert salida.startswith("ERROR: no existe ningún cliente con cliente_id=999.")


def test_ultimo_pedido_y_suma_coherente() -> None:
    ultimo = json.loads(buscar_ultimo_pedido.invoke({"cliente_id": 102}))
    assert ultimo["pedido_id"] == "P-1023" and ultimo["monto"] == 6200
    assert ultimo["monto"] <= PEDIDOS_DB[102]["total"]
    assert buscar_ultimo_pedido.invoke({"cliente_id": 7}).startswith("ERROR: el cliente_id=7 no existe")


@pytest.mark.parametrize("cliente_id", [0, -5, "abc"])
def test_pydantic_rechaza_cliente_id_invalido(cliente_id: object) -> None:
    with pytest.raises(ValidationError):
        buscar_pedidos.invoke({"cliente_id": cliente_id})


def test_pydantic_acepta_id_numerico_como_texto() -> None:
    assert json.loads(buscar_pedidos.invoke({"cliente_id": "205"}))["pedidos"] == 1


def test_docstrings_describen_cuando_usar_cada_herramienta() -> None:
    descripciones = {h.name: h.description for h in herramientas}
    assert set(descripciones) == {"buscar_cliente_por_nombre", "buscar_pedidos", "buscar_ultimo_pedido"}
    for descripcion in descripciones.values():
        assert len(descripcion) > 300
        assert "ERROR" in descripcion
    assert "SIEMPRE que el usuario mencione un cliente por su nombre" in descripciones["buscar_cliente_por_nombre"]
    assert "primero" in descripciones["buscar_pedidos"] and "buscar_cliente_por_nombre" in descripciones["buscar_pedidos"]
    assert "¿y el último?" in descripciones["buscar_ultimo_pedido"]


def test_esquema_de_argumentos_llega_al_llm() -> None:
    argumento = buscar_pedidos.args["cliente_id"]
    assert argumento["type"] == "integer"
    assert argumento["exclusiveMinimum"] == 0
    assert "No es el nombre" in argumento["description"]

