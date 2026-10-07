import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

import agente
from agente import AppAgente, grafo, preguntar, recortar_historial
from errors import LimiteRecursionError
from tests.modelo_local import ModeloEnBucle, ModeloPedidosLocal, llamada


def _hilo() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def _llamadas(mensajes: list[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
    return [(tc["name"], tc["args"]) for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls]


def test_state_graph_usa_messages_state() -> None:
    assert grafo.state_schema is MessagesState
    assert "messages" in MessagesState.__annotations__


def test_nodos_y_aristas_del_ciclo(app: AppAgente) -> None:
    dibujo = app.get_graph()
    assert {"__start__", "modelo", "herramientas", "__end__"} <= set(dibujo.nodes)
    aristas = {(a.source, a.target, a.conditional) for a in dibujo.edges}
    assert ("__start__", "modelo", False) in aristas
    assert ("modelo", "herramientas", True) in aristas
    assert ("modelo", "__end__", True) in aristas
    assert ("herramientas", "modelo", False) in aristas


def test_arista_condicional_es_tools_condition() -> None:
    rama = grafo.branches["modelo"]["tools_condition"]
    assert rama.ends == {"tools": "herramientas", "__end__": "__end__"}


def test_nodo_herramientas_tiene_las_tres_tools_y_maneja_errores() -> None:
    nodo = grafo.nodes["herramientas"].runnable
    assert isinstance(nodo, ToolNode)
    assert set(nodo.tools_by_name) == {"buscar_cliente_por_nombre", "buscar_pedidos", "buscar_ultimo_pedido"}


async def test_multi_paso_llama_dos_herramientas_en_orden(app: AppAgente) -> None:
    salida = await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    tipos = [type(m).__name__ for m in salida["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert _llamadas(salida["messages"]) == [
        ("buscar_cliente_por_nombre", {"nombre": "Ana García"}),
        ("buscar_pedidos", {"cliente_id": 102}),
    ]
    assert salida["messages"][-1].content == "Ana García tuvo 3 pedidos por un total de $14.500."


async def test_id_directo_usa_una_sola_herramienta(app: AppAgente) -> None:
    salida = await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?")
    assert _llamadas(salida["messages"]) == [("buscar_pedidos", {"cliente_id": 102})]
    assert "3 pedidos" in salida["messages"][-1].content and "$14.500" in salida["messages"][-1].content


async def test_nombre_incompleto_hace_segundo_intento(app: AppAgente) -> None:
    salida = await preguntar(app, _hilo(), "¿Cuántos pedidos tiene García?")
    assert _llamadas(salida["messages"]) == [
        ("buscar_cliente_por_nombre", {"nombre": "García"}),
        ("buscar_cliente_por_nombre", {"nombre": "Ana García"}),
        ("buscar_pedidos", {"cliente_id": 102}),
    ]
    assert str(salida["messages"][2].content).startswith("ERROR: 'García' es un nombre incompleto")
    assert salida["messages"][-1].content == "Ana García tuvo 3 pedidos por un total de $14.500."


async def test_cliente_inexistente_pide_aclaracion_sin_inventar(app: AppAgente) -> None:
    salida = await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?")
    assert _llamadas(salida["messages"]) == [("buscar_cliente_por_nombre", {"nombre": "Roberto Sánchez"})]
    respuesta = str(salida["messages"][-1].content)
    assert "¿Podés confirmar el nombre completo" in respuesta
    assert "$" not in respuesta


async def test_argumento_invalido_vuelve_como_tool_message_de_error(
    app: AppAgente, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PideIdInvalido(ModeloPedidosLocal):
        def _decidir(self, mensajes: list[BaseMessage]) -> AIMessage:
            return llamada("buscar_pedidos", cliente_id=0)

    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: PideIdInvalido())
    salida = await preguntar(app, _hilo(), "Pedidos del cliente cero")
    observacion = salida["messages"][2]
    assert isinstance(observacion, ToolMessage)
    assert observacion.status == "error"
    assert "greater than 0" in str(observacion.content)
    assert salida["messages"][-1].content == "No pude hacer la consulta con esos datos. ¿Me confirmás el cliente?"


async def test_el_modelo_recibe_instrucciones_y_el_historial(app: AppAgente, modelo_local: ModeloPedidosLocal) -> None:
    await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    assert len(modelo_local.llamadas) == 3
    primera, ultima = modelo_local.llamadas[0], modelo_local.llamadas[-1]
    assert isinstance(primera[0], SystemMessage)
    assert [type(m).__name__ for m in ultima[1:]] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage",
    ]


async def test_recursion_limit_corta_el_bucle(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    bucle = ModeloEnBucle()
    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: bucle)
    with pytest.raises(LimiteRecursionError, match=r"^recursion_limit: el agente llegó a 10 pasos sin una respuesta final"):
        await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo el cliente 102?")
    assert bucle.llamadas == 5


async def test_recursion_limit_de_la_llamada_llega_al_grafo(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    bucle = ModeloEnBucle()
    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: bucle)
    with pytest.raises(LimiteRecursionError, match="llegó a 4 pasos"):
        await preguntar(app, _hilo(), "¿Cuántos pedidos tuvo el cliente 102?", recursion_limit=4)
    assert bucle.llamadas == 2


def test_recortar_historial_respeta_el_tope_y_empieza_en_humano() -> None:
    mensajes: list[BaseMessage] = []
    for i in range(10):
        pedido = llamada("buscar_pedidos", cliente_id=102)
        mensajes += [HumanMessage(content=f"p{i}"), pedido]
        mensajes += [ToolMessage(content="{}", tool_call_id=pedido.tool_calls[0]["id"]), AIMessage(content=f"r{i}")]
    recortados = recortar_historial(mensajes, max_mensajes=6)
    assert len(recortados) <= 6
    assert isinstance(recortados[0], HumanMessage)
    assert recortados[-1].content == "r9"
    assert recortar_historial(mensajes, max_mensajes=100) == mensajes


def test_recortar_historial_si_el_turno_supera_el_tope() -> None:
    mensajes: list[BaseMessage] = [HumanMessage(content="viejo"), AIMessage(content="ok"), HumanMessage(content="nuevo")]
    mensajes += [llamada("buscar_pedidos", cliente_id=102), ToolMessage(content="{}", tool_call_id="x")]
    assert [m.content for m in recortar_historial(mensajes, max_mensajes=2)][0] == "nuevo"


async def test_max_mensajes_de_la_llamada_llega_al_modelo(app: AppAgente, modelo_local: ModeloPedidosLocal) -> None:
    hilo = _hilo()
    for _ in range(3):
        await preguntar(app, hilo, "¿Cuántos pedidos tuvo el cliente 102?", max_mensajes=4)
    ultima = modelo_local.llamadas[-1]
    assert len(ultima) - 1 <= 4
    assert isinstance(ultima[1], HumanMessage)
