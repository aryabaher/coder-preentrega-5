"""Base de datos simulada de clientes y pedidos, y las herramientas que la consultan."""

import json
import unicodedata

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Dos tablas separadas: el usuario nombra al cliente y los pedidos se buscan por ID,
# así que el agente tiene que encadenar dos herramientas para responder.
CLIENTES_DB: dict[str, int] = {
    "ana garcía": 102,
    "juan pérez": 205,
    "maría lópez": 310,
}

PEDIDOS_DB: dict[int, dict[str, int]] = {
    102: {"pedidos": 3, "total": 14500},
    205: {"pedidos": 1, "total": 3200},
    310: {"pedidos": 5, "total": 27800},
}

ULTIMO_PEDIDO_DB: dict[int, dict[str, str | int]] = {
    102: {"pedido_id": "P-1023", "fecha": "2026-09-28", "monto": 6200, "estado": "entregado"},
    205: {"pedido_id": "P-2051", "fecha": "2026-08-14", "monto": 3200, "estado": "entregado"},
    310: {"pedido_id": "P-3105", "fecha": "2026-10-02", "monto": 4100, "estado": "en camino"},
}


class BuscarClienteInput(BaseModel):
    nombre: str = Field(
        description="Nombre y apellido del cliente tal como lo dijo el usuario, por ejemplo 'Ana García'."
    )


class ClienteIdInput(BaseModel):
    cliente_id: int = Field(
        gt=0,
        description="ID numérico interno del cliente, por ejemplo 102. No es el nombre.",
    )


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


_CLIENTES_NORMALIZADOS: dict[str, tuple[str, int]] = {
    _normalizar(nombre): (nombre.title(), cliente_id) for nombre, cliente_id in CLIENTES_DB.items()
}


@tool(args_schema=BuscarClienteInput)
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.

    Usar esta herramienta SIEMPRE que el usuario mencione un cliente por su nombre
    y necesites averiguar su cliente_id antes de poder consultar sus pedidos.
    Si el usuario dio solo el apellido o solo el nombre (por ejemplo "García"),
    llamarla igual con eso, sin pedir antes el nombre completo: la herramienta
    responde con los nombres completos que coinciden.
    No usarla si el usuario ya dio el cliente_id numérico: en ese caso llamar
    directo a buscar_pedidos o a buscar_ultimo_pedido.

    Args:
        nombre: nombre del cliente tal como lo dio el usuario, completo o
            parcial (por ejemplo "Ana García" o "García"). No distingue
            mayúsculas ni tildes.

    Devuelve un JSON {"nombre": ..., "cliente_id": ...} si el cliente existe.
    Devuelve un texto que empieza con ERROR en tres casos:
    - nombre vacío: pedirle al usuario el nombre del cliente;
    - nombre incompleto (solo el apellido o solo el nombre): el ERROR trae los
      nombres completos que coinciden; volver a llamar a esta herramienta con
      ese nombre completo, o preguntar al usuario si hay más de uno;
    - ningún cliente con ese nombre: no inventar datos; pedirle al usuario que
      confirme el nombre completo o el cliente_id.
    """
    clave = _normalizar(nombre)
    if not clave:
        return "ERROR: el nombre está vacío. Pedile al usuario el nombre y apellido del cliente."
    if clave in _CLIENTES_NORMALIZADOS:
        nombre_registrado, cliente_id = _CLIENTES_NORMALIZADOS[clave]
        # La observación del rol tool es siempre str: JSON con los datos o "ERROR: ..." con cómo seguir.
        return json.dumps({"nombre": nombre_registrado, "cliente_id": cliente_id}, ensure_ascii=False)
    palabras = clave.split()
    candidatos = [
        nombre_registrado
        for normalizado, (nombre_registrado, _) in _CLIENTES_NORMALIZADOS.items()
        if all(palabra in normalizado.split() for palabra in palabras)
    ]
    if candidatos:
        return (
            f"ERROR: '{nombre.strip()}' es un nombre incompleto. Coincide con: {', '.join(candidatos)}. "
            "Volvé a llamar a buscar_cliente_por_nombre con el nombre completo; "
            "si hay más de una coincidencia, preguntale al usuario cuál es."
        )
    return (
        f"ERROR: no se encontró ningún cliente con el nombre '{nombre.strip()}'. "
        "No inventes datos: pedile al usuario que confirme el nombre completo o el cliente_id."
    )


@tool(args_schema=ClienteIdInput)
def buscar_pedidos(cliente_id: int) -> str:
    """Busca la cantidad de pedidos y el monto total gastado por un cliente,
    dado su cliente_id NUMÉRICO (no su nombre: si solo tenés el nombre, primero
    usá buscar_cliente_por_nombre para obtener el ID).

    Usarla para preguntas de cantidad de pedidos, total gastado o resumen de compras.
    No sirve para el detalle del último pedido: para eso está buscar_ultimo_pedido.

    Args:
        cliente_id: ID numérico del cliente, mayor que 0 (por ejemplo 102).

    Devuelve un JSON {"cliente_id": ..., "pedidos": ..., "total": ...}; el total
    está en pesos, sin decimales. Devuelve un texto que empieza con ERROR si el
    cliente_id no existe en la base de datos.
    """
    if cliente_id not in PEDIDOS_DB:
        return (
            f"ERROR: no existe ningún cliente con cliente_id={cliente_id}. "
            "Si tenés el nombre, buscá el ID con buscar_cliente_por_nombre; si no, pedile al usuario el ID correcto."
        )
    return json.dumps({"cliente_id": cliente_id, **PEDIDOS_DB[cliente_id]})


@tool(args_schema=ClienteIdInput)
def buscar_ultimo_pedido(cliente_id: int) -> str:
    """Busca el detalle del pedido más reciente de un cliente, dado su cliente_id NUMÉRICO.

    Usarla cuando el usuario pregunte por el último pedido, el más reciente, su
    fecha, su monto o su estado (por ejemplo "¿y el último?" después de hablar de
    un cliente). Si solo tenés el nombre, primero usá buscar_cliente_por_nombre.
    No sirve para el total histórico: para eso está buscar_pedidos.

    Args:
        cliente_id: ID numérico del cliente, mayor que 0 (por ejemplo 102).

    Devuelve un JSON {"cliente_id", "pedido_id", "fecha", "monto", "estado"};
    la fecha es AAAA-MM-DD y el monto está en pesos. Devuelve un texto que empieza
    con ERROR si el cliente_id no existe o no tiene pedidos.
    """
    if cliente_id not in ULTIMO_PEDIDO_DB:
        return (
            f"ERROR: el cliente_id={cliente_id} no existe o no tiene pedidos. "
            "Confirmá el ID con buscar_cliente_por_nombre o pedile al usuario el dato correcto."
        )
    return json.dumps({"cliente_id": cliente_id, **ULTIMO_PEDIDO_DB[cliente_id]})


herramientas = [buscar_cliente_por_nombre, buscar_pedidos, buscar_ultimo_pedido]
