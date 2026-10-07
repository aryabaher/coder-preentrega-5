"""Traza ReAct de cada turno: JSON (mensaje por mensaje) y log legible."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

type Traza = list[dict[str, Any]]

RUTA_TRAZA_JSON = Path("traza_ejecucion.json")
RUTA_TRAZA_LOG = Path("traza_ejecucion.log")


def extraer_texto(mensaje: BaseMessage) -> str:
    """Normaliza el content de un mensaje (Anthropic puede devolver una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))


def mensajes_del_turno(mensajes: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Corta el historial del thread desde la última pregunta del usuario."""
    for indice in range(len(mensajes) - 1, -1, -1):
        if isinstance(mensajes[indice], HumanMessage):
            return list(mensajes[indice:])
    return list(mensajes)


def serializar_traza(mensajes: Sequence[BaseMessage]) -> Traza:
    """Convierte los mensajes de LangGraph a un formato JSON simple."""
    traza: Traza = []
    for m in mensajes:
        entrada: dict[str, Any] = {"tipo": type(m).__name__, "contenido": extraer_texto(m)}
        if isinstance(m, AIMessage) and m.tool_calls:
            entrada["tool_calls"] = [
                {"nombre": tc["name"], "argumentos": tc["args"]} for tc in m.tool_calls
            ]
        if isinstance(m, ToolMessage):
            entrada["herramienta"] = m.name
            entrada["status"] = m.status
        traza.append(entrada)
    return traza


def herramientas_invocadas(mensajes: Sequence[BaseMessage]) -> list[str]:
    """Llamadas a herramientas en orden, por ejemplo ['buscar_cliente_por_nombre(nombre='Ana García')']."""
    llamadas: list[str] = []
    for m in mensajes:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                argumentos = ", ".join(f"{clave}={valor!r}" for clave, valor in tc["args"].items())
                llamadas.append(f"{tc['name']}({argumentos})")
    return llamadas


def lineas_react(mensajes: Sequence[BaseMessage]) -> list[str]:
    """Traza legible del ciclo: decisión del agente, observación de la herramienta y respuesta."""
    lineas: list[str] = []
    for anterior, m in zip([None, *mensajes], mensajes):
        fallo = isinstance(anterior, ToolMessage) and (
            anterior.status == "error" or extraer_texto(anterior).startswith("ERROR")
        )
        if isinstance(m, HumanMessage):
            lineas.append(f'Usuario: "{extraer_texto(m)}"')
        elif isinstance(m, AIMessage) and m.tool_calls:
            if fallo:
                lineas.append("→ El agente razona: la herramienta devolvió ERROR → hace un segundo intento.")
            elif isinstance(anterior, ToolMessage):
                lineas.append("→ El agente razona: todavía le falta un dato → llama otra herramienta.")
            for llamada in herramientas_invocadas([m]):
                lineas.append(f"→ El agente decide usar la herramienta: {llamada}")
        elif isinstance(m, ToolMessage):
            lineas.append(f"→ La herramienta devuelve: {extraer_texto(m)}")
        elif isinstance(m, AIMessage):
            if fallo:
                lineas.append("→ El agente razona: la herramienta devolvió ERROR → pide una aclaración, no inventa.")
            elif isinstance(anterior, ToolMessage):
                lineas.append("→ El agente razona: ya tiene los datos → responde.")
            else:
                lineas.append("→ El agente razona: no necesita herramientas o le falta contexto → responde.")
            lineas.append(f'Respuesta: "{extraer_texto(m)}"')
    return lineas


def registrar_turno(thread_id: str, pregunta: str, mensajes: Sequence[BaseMessage]) -> dict[str, Any]:
    """Arma la entrada de un turno: los mensajes nuevos, las llamadas y la respuesta final."""
    turno = mensajes_del_turno(mensajes)
    return {
        "thread_id": thread_id,
        "pregunta": pregunta,
        "mensajes_en_checkpoint": len(mensajes),
        "herramientas_invocadas": herramientas_invocadas(turno),
        "respuesta": extraer_texto(turno[-1]),
        "traza_react": lineas_react(turno),
        "mensajes": serializar_traza(turno),
    }


def guardar_traza(
    traza_completa: dict[str, Any],
    ruta_json: Path = RUTA_TRAZA_JSON,
    ruta_log: Path = RUTA_TRAZA_LOG,
) -> None:
    """Escribe la traza en JSON y el mismo recorrido como log de texto."""
    ruta_json.write_text(json.dumps(traza_completa, ensure_ascii=False, indent=2), encoding="utf-8")
    bloques = [
        f"## {nombre} (thread_id={turno['thread_id']})\n" + "\n".join(turno["traza_react"])
        for nombre, turno in traza_completa["turnos"].items()
    ]
    ruta_log.write_text("\n\n".join(bloques) + "\n", encoding="utf-8")
