"""Dobles del LLM vinculado a las herramientas. Leen el historial que manda el grafo; no usan red."""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

_NOMBRE = re.compile(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*")
_NO_NOMBRES = {"Cuántos", "Cuál", "Cuánto", "Y"}


def llamada(herramienta: str, /, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": herramienta, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}],
    )


def pesos(monto: int) -> str:
    return f"${monto:,}".replace(",", ".")


def _ultimo_humano(mensajes: Sequence[BaseMessage]) -> str:
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, HumanMessage):
            return str(mensaje.content)
    return ""


def _pide_ultimo(mensajes: Sequence[BaseMessage]) -> bool:
    """La pregunta actual pide el último pedido, o es un "¿Y <cliente>?" después de pedirlo."""
    preguntas = [str(m.content).lower() for m in mensajes if isinstance(m, HumanMessage)]
    if not preguntas:
        return False
    if "último" in preguntas[-1] or "ultimo" in preguntas[-1]:
        return True
    seguimiento = preguntas[-1].lstrip("¿ ").startswith("y ")
    return seguimiento and len(preguntas) > 1 and "último" in preguntas[-2]


def _nombre_en(texto: str) -> str | None:
    for candidato in _NOMBRE.findall(texto):
        palabras = [p for p in candidato.split() if p not in _NO_NOMBRES]
        if palabras:
            return " ".join(palabras)
    return None


def _datos(mensaje: ToolMessage) -> dict[str, Any] | None:
    try:
        datos = json.loads(str(mensaje.content))
    except json.JSONDecodeError:
        return None
    return datos if isinstance(datos, dict) else None


def _ultimo_cliente(mensajes: Sequence[BaseMessage]) -> dict[str, Any] | None:
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, ToolMessage) and (datos := _datos(mensaje)) and "cliente_id" in datos:
            return datos
    return None


def _nombre_de(mensajes: Sequence[BaseMessage], cliente_id: int) -> str:
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, ToolMessage) and (datos := _datos(mensaje)) and datos.get("cliente_id") == cliente_id:
            if "nombre" in datos:
                return str(datos["nombre"])
    return f"el cliente {cliente_id}"


class ModeloPedidosLocal:
    """Simula las decisiones del LLM en el ciclo ReAct. Guarda cada lista de mensajes recibida."""

    def __init__(self) -> None:
        self.llamadas: list[list[BaseMessage]] = []

    async def ainvoke(self, mensajes: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        self.llamadas.append(list(mensajes))
        ultimo = mensajes[-1]
        if isinstance(ultimo, ToolMessage):
            return self._observar(mensajes, ultimo)
        return self._decidir(mensajes)

    def _decidir(self, mensajes: list[BaseMessage]) -> AIMessage:
        pregunta = _ultimo_humano(mensajes)
        nombre = _nombre_en(pregunta)
        if nombre:
            return llamada("buscar_cliente_por_nombre", nombre=nombre)
        if id_explicito := re.search(r"cliente (\d+)", pregunta):
            return llamada("buscar_pedidos", cliente_id=int(id_explicito.group(1)))
        if "último" in pregunta.lower() or "ultimo" in pregunta.lower():
            cliente = _ultimo_cliente(mensajes[:-1])
            if cliente is None:
                return AIMessage(content="¿De qué cliente querés ver el último pedido? Pasame el nombre o el cliente_id.")
            return llamada("buscar_ultimo_pedido", cliente_id=cliente["cliente_id"])
        return AIMessage(content="Puedo consultar pedidos de clientes. ¿De qué cliente se trata?")

    def _observar(self, mensajes: list[BaseMessage], observacion: ToolMessage) -> AIMessage:
        texto = str(observacion.content)
        if observacion.status == "error":
            return AIMessage(content="No pude hacer la consulta con esos datos. ¿Me confirmás el cliente?")
        if texto.startswith("ERROR"):
            coincide = re.search(r"Coincide con: ([^.,]+)\.", texto)
            ya_reintento = sum(isinstance(m, ToolMessage) and str(m.content).startswith("ERROR") for m in mensajes) > 1
            if coincide and not ya_reintento:
                return llamada("buscar_cliente_por_nombre", nombre=coincide.group(1).strip())
            nombre = re.search(r"'([^']+)'", texto)
            dato = nombre.group(1) if nombre else "ese cliente"
            return AIMessage(
                content=f"No encontré a un cliente llamado {dato}. ¿Podés confirmar el nombre completo o pasarme el cliente_id?"
            )
        datos = _datos(observacion) or {}
        if observacion.name == "buscar_cliente_por_nombre":
            if _pide_ultimo(mensajes):
                return llamada("buscar_ultimo_pedido", cliente_id=datos["cliente_id"])
            return llamada("buscar_pedidos", cliente_id=datos["cliente_id"])
        nombre_cliente = _nombre_de(mensajes, datos["cliente_id"])
        if observacion.name == "buscar_pedidos":
            pedidos = "1 pedido" if datos["pedidos"] == 1 else f"{datos['pedidos']} pedidos"
            return AIMessage(content=f"{nombre_cliente} tuvo {pedidos} por un total de {pesos(datos['total'])}.")
        return AIMessage(
            content=(
                f"El último pedido de {nombre_cliente} es {datos['pedido_id']}, del {datos['fecha']}, "
                f"por {pesos(datos['monto'])} ({datos['estado']})."
            )
        )


class ModeloEnBucle:
    """Siempre pide otra herramienta: sirve para probar recursion_limit."""

    def __init__(self) -> None:
        self.llamadas = 0

    async def ainvoke(self, mensajes: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        self.llamadas += 1
        return llamada("buscar_pedidos", cliente_id=102)


class ModeloConFallos:
    """Lanza las excepciones dadas, en orden, y después responde."""

    def __init__(self, *fallos: BaseException, respuesta: AIMessage | None = None) -> None:
        self.fallos = list(fallos)
        self.llamadas = 0
        self.respuesta = respuesta or AIMessage(content="Listo.")

    async def ainvoke(self, mensajes: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        self.llamadas += 1
        if self.fallos:
            raise self.fallos.pop(0)
        return self.respuesta
