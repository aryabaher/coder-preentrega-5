"""Agente ReAct: StateGraph sobre MessagesState, nodo modelo + nodo herramientas y checkpoint SQLite."""

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from errors import AgenteError, ConsultaInvalidaError, LimiteRecursionError, SalidaTruncadaError
from herramientas import herramientas
from modelo import clasificar_error, crear_llm

RUTA_DB = "checkpoints.sqlite"
RECURSION_LIMIT = 10
MAX_MENSAJES = 20
MAX_INTENTOS = 3
ESPERA_INICIAL_S = 0.5

INSTRUCCIONES = (
    "Sos un asistente que responde consultas sobre clientes y sus pedidos. Respondé en español. "
    "Los datos salen solo de las herramientas: nunca inventes cantidades, montos ni IDs. "
    "Si la pregunta se apoya en algo dicho antes en la conversación, usá ese contexto. "
    "Si una herramienta devuelve ERROR, seguí la indicación del mensaje. "
    "Mostrá los montos en pesos con punto de miles, por ejemplo $14.500."
)

type AppAgente = CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]


async def nodo_modelo(state: MessagesState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Le pasa el historial al LLM y devuelve su respuesta.

    La respuesta puede traer tool_calls (tools_condition la manda a "herramientas") o
    ser la respuesta final (va a END). El LLM decide solo: no hay rutas if/else por
    palabra clave. 429/cuota y red/timeout se reintentan con backoff (0.5s, 1s); una
    401 no. Una respuesta cortada por max_tokens se rechaza para no guardar media
    respuesta en el checkpoint.
    """
    max_mensajes = config.get("configurable", {}).get("max_mensajes", MAX_MENSAJES)
    mensajes = [SystemMessage(content=INSTRUCCIONES), *recortar_historial(state["messages"], max_mensajes)]
    ultimo_error: AgenteError | None = None
    for intento in range(MAX_INTENTOS):
        if intento:
            await asyncio.sleep(ESPERA_INICIAL_S * 2 ** (intento - 1))
        try:
            respuesta = await obtener_llm_con_herramientas().ainvoke(mensajes)
        except Exception as exc:
            ultimo_error = clasificar_error(exc)
            if ultimo_error is exc:
                raise
            if not ultimo_error.reintentable:
                raise ultimo_error from exc
            continue
        if _truncada(respuesta):
            raise SalidaTruncadaError(
                "Salida truncada: el modelo cortó la respuesta por max_tokens. No se guarda a medias: "
                "subí MAX_TOKENS en modelo.py o pedí una respuesta más corta."
            )
        return {"messages": [respuesta]}
    assert ultimo_error is not None
    raise type(ultimo_error)(
        f"{ultimo_error} (reintentos agotados: {MAX_INTENTOS} intentos; probá de nuevo en unos segundos)"
    )


grafo = StateGraph(MessagesState)  # hereda el reducer add_messages: cada nodo suma mensajes al historial
grafo.add_node("modelo", nodo_modelo)
grafo.add_node("herramientas", ToolNode(herramientas, handle_tool_errors=True))

grafo.add_edge(START, "modelo")
grafo.add_conditional_edges(
    "modelo",
    tools_condition,  # mira si el último mensaje trae tool_calls
    {"tools": "herramientas", END: END},
)
# herramientas -> modelo cierra el ciclo ReAct: la observación vuelve al modelo,
# que decide si llama otra herramienta (o la misma con otros argumentos) o responde.
grafo.add_edge("herramientas", "modelo")


@asynccontextmanager
async def abrir_agente(ruta_db: str = RUTA_DB) -> AsyncIterator[AppAgente]:
    """Compila el grafo con un AsyncSqliteSaver sobre `ruta_db` y lo cierra al salir."""
    async with AsyncSqliteSaver.from_conn_string(ruta_db) as checkpointer:
        await checkpointer.setup()
        yield grafo.compile(checkpointer=checkpointer)


async def preguntar(
    app: AppAgente,
    thread_id: str,
    mensaje: str,
    *,
    recursion_limit: int = RECURSION_LIMIT,
    max_mensajes: int = MAX_MENSAJES,
) -> dict[str, Any]:
    """Manda un turno del usuario al agente y devuelve el estado final del thread.

    Args:
        app: grafo compilado con checkpointer (ver `abrir_agente`).
        thread_id: conversación. El mismo thread_id recupera el historial previo.
        mensaje: pregunta del usuario. No puede estar vacía.
        recursion_limit: techo de pasos del grafo (default 10, máximo 50).
        max_mensajes: cuántos mensajes del historial ve el LLM en cada paso.

    Raises:
        ConsultaInvalidaError: mensaje vacío, thread_id con caracteres raros o límites fuera de rango.
        LimiteRecursionError: el agente no llegó a una respuesta en `recursion_limit` pasos.
        AgenteError: 401/key, 429/cuota, red/timeout o salida truncada, desde el nodo modelo.
    """
    try:
        consulta = Consulta(
            thread_id=thread_id,
            mensaje=mensaje,
            recursion_limit=recursion_limit,
            max_mensajes=max_mensajes,
        )
    except ValidationError as exc:
        campos = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ConsultaInvalidaError(f"Consulta inválida: {campos}") from exc
    config = crear_config(consulta.thread_id, consulta.recursion_limit, consulta.max_mensajes)
    try:
        salida: dict[str, Any] = await app.ainvoke(
            {"messages": [HumanMessage(content=consulta.mensaje)]}, config=config
        )
    except GraphRecursionError as exc:
        raise LimiteRecursionError(
            f"recursion_limit: el agente llegó a {consulta.recursion_limit} pasos sin una respuesta final. "
            "Se cortó para no seguir gastando llamadas: reformulá la pregunta o revisá los docstrings de las herramientas."
        ) from exc
    return salida


def crear_config(
    thread_id: str, recursion_limit: int = RECURSION_LIMIT, max_mensajes: int = MAX_MENSAJES
) -> RunnableConfig:
    """Config de una invocación: el thread_id elige la conversación del checkpoint."""
    return {
        "configurable": {"thread_id": thread_id, "max_mensajes": max_mensajes},
        "recursion_limit": recursion_limit,
    }


class Consulta(BaseModel):
    """Entrada validada de un turno del usuario."""

    model_config = ConfigDict(str_strip_whitespace=True)

    thread_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    mensaje: str = Field(min_length=1, max_length=4000)
    recursion_limit: int = Field(default=RECURSION_LIMIT, ge=1, le=50)
    max_mensajes: int = Field(default=MAX_MENSAJES, ge=2, le=200)


@cache
def obtener_llm_con_herramientas() -> Runnable[LanguageModelInput, BaseMessage]:
    """Crea el LLM la primera vez que el grafo lo necesita y lo vincula a las herramientas.

    Se crea en la primera llamada y no al importar, para que una clave faltante salga
    como el 401/key del agente y no como un error del SDK al cargar el módulo.
    """
    llm = crear_llm()
    llm_con_herramientas = llm.bind_tools(herramientas)
    return llm_con_herramientas


def recortar_historial(mensajes: Sequence[BaseMessage], max_mensajes: int = MAX_MENSAJES) -> list[BaseMessage]:
    """Devuelve los últimos `max_mensajes` del historial, empezando en un mensaje del usuario.

    El checkpoint guarda el historial completo; esto solo limita lo que se le manda al
    LLM, para que una conversación larga no crezca en tokens sin techo. Empezar en un
    HumanMessage evita mandar un ToolMessage sin el AIMessage que lo pidió.
    """
    recortados: list[BaseMessage] = trim_messages(
        mensajes,
        max_tokens=max_mensajes,
        token_counter=len,
        strategy="last",
        start_on="human",
    )
    if recortados:
        return recortados
    # El turno actual solo ya supera el tope: se manda completo desde la última pregunta.
    for indice in range(len(mensajes) - 1, -1, -1):
        if isinstance(mensajes[indice], HumanMessage):
            return list(mensajes[indice:])
    return list(mensajes)


def _truncada(respuesta: BaseMessage) -> bool:
    metadata = respuesta.response_metadata
    return metadata.get("finish_reason") == "length" or metadata.get("stop_reason") == "max_tokens"


async def borrar_hilos(app: AppAgente, thread_ids: Iterable[str]) -> None:
    """Borra del checkpoint los thread_id dados. Sirve para que una demo arranque de cero."""
    checkpointer = app.checkpointer
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise AgenteError("El grafo no tiene checkpointer: compilarlo con abrir_agente().")
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
