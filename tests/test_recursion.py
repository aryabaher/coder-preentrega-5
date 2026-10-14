import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

import experimento_recursion
from agente import AppAgente, preguntar
from errors import LimiteRecursionError
from tests.modelo_local import ModeloPedidosLocal

GARCIA = "¿Cuántos pedidos tiene García?"


def _sin_llamadas_huerfanas(mensajes: list[BaseMessage]) -> bool:
    pedidas = {tc["id"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls}
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    return pedidas == respondidas


async def test_recursion_limit_5_corta_el_caso_garcia(app: AppAgente, modelo_local: ModeloPedidosLocal) -> None:
    with pytest.raises(LimiteRecursionError, match="llegó a 5 pasos"):
        await preguntar(app, "garcia-5", GARCIA, recursion_limit=5)
    assert len(modelo_local.llamadas) == 3
    estado = await app.aget_state({"configurable": {"thread_id": "garcia-5"}})
    mensajes = estado.values["messages"]
    assert isinstance(mensajes[-2], AIMessage) and mensajes[-2].tool_calls[0]["name"] == "buscar_pedidos"
    assert isinstance(mensajes[-1], ToolMessage) and mensajes[-1].status == "error"
    assert mensajes[-1].content == "ERROR: no se ejecutó porque el turno se cortó por recursion_limit."
    assert _sin_llamadas_huerfanas(mensajes)


async def test_recursion_limit_7_alcanza_para_garcia(app: AppAgente) -> None:
    salida = await preguntar(app, "garcia-7", GARCIA, recursion_limit=7)
    assert salida["messages"][-1].content == "Ana García tuvo 3 pedidos por un total de $14.500."


async def test_limite_igual_a_los_pasos_devuelve_la_respuesta_guardada(app: AppAgente) -> None:
    """Ana usa 5 nodos: LangGraph corta en la vuelta siguiente, con la respuesta ya en el checkpoint."""
    salida = await preguntar(app, "ana-5", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?", recursion_limit=5)
    assert salida["messages"][-1].content == "Ana García tuvo 3 pedidos por un total de $14.500."
    with pytest.raises(LimiteRecursionError, match="llegó a 4 pasos"):
        await preguntar(app, "ana-4", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?", recursion_limit=4)


async def test_el_thread_sigue_usable_despues_del_corte(app: AppAgente) -> None:
    with pytest.raises(LimiteRecursionError):
        await preguntar(app, "garcia-retoma", GARCIA, recursion_limit=5)
    salida = await preguntar(app, "garcia-retoma", GARCIA)
    assert salida["messages"][-1].content == "Ana García tuvo 3 pedidos por un total de $14.500."
    assert _sin_llamadas_huerfanas(salida["messages"])
    estado = await app.aget_state({"configurable": {"thread_id": "garcia-retoma"}})
    assert estado.next == ()


async def test_corte_despues_de_herramientas_no_agrega_cierres(app: AppAgente) -> None:
    with pytest.raises(LimiteRecursionError, match="llegó a 4 pasos"):
        await preguntar(app, "garcia-4", GARCIA, recursion_limit=4)
    estado = await app.aget_state({"configurable": {"thread_id": "garcia-4"}})
    cierres = [m for m in estado.values["messages"] if "recursion_limit" in str(m.content)]
    assert cierres == []
    assert _sin_llamadas_huerfanas(estado.values["messages"])


async def test_experimento_con_los_tres_limites(ruta_db: str, capsys: pytest.CaptureFixture[str]) -> None:
    resultados = await experimento_recursion.probar_limites((5, 7, 10), ruta_db)
    assert resultados == {5: "cortó", 7: "respondió", 10: "respondió"}
    salida = capsys.readouterr().out
    assert "## recursion_limit=5 (thread_id=experimento-recursion-5)" in salida
    assert "Error controlado: recursion_limit: el agente llegó a 5 pasos" in salida
    assert "# Mismo thread_id, de nuevo con recursion_limit=10" in salida
