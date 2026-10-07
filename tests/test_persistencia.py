from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agente import AppAgente, abrir_agente, borrar_hilos, crear_config, preguntar


def _llamadas(mensajes: list[BaseMessage]) -> list[str]:
    return [tc["name"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls]


def test_crear_config_lleva_thread_id_y_recursion_limit() -> None:
    assert crear_config("conversacion-demo-1") == {
        "configurable": {"thread_id": "conversacion-demo-1", "max_mensajes": 20},
        "recursion_limit": 10,
    }
    assert crear_config("x", recursion_limit=6, max_mensajes=8)["recursion_limit"] == 6


async def test_checkpointer_es_async_sqlite_saver(app: AppAgente) -> None:
    assert isinstance(app.checkpointer, AsyncSqliteSaver)


async def test_mismo_thread_id_recuerda_el_cliente(app: AppAgente) -> None:
    await preguntar(app, "sesion-memoria", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    salida = await preguntar(app, "sesion-memoria", "¿Y el último?")
    assert _llamadas(salida["messages"])[-1] == "buscar_ultimo_pedido"
    assert salida["messages"][-1].content == (
        "El último pedido de Ana García es P-1023, del 2026-09-28, por $6.200 (entregado)."
    )
    estado = await app.aget_state({"configurable": {"thread_id": "sesion-memoria"}})
    assert len(estado.values["messages"]) == len(salida["messages"]) == 10


async def test_seguimiento_con_otro_cliente_conserva_la_intencion(app: AppAgente) -> None:
    await preguntar(app, "sesion-intencion", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    salida = await preguntar(app, "sesion-intencion", "¿Y Juan Pérez?")
    assert salida["messages"][-1].content == "Juan Pérez tuvo 1 pedido por un total de $3.200."


async def test_otro_thread_id_no_ve_el_historial(app: AppAgente) -> None:
    await preguntar(app, "sesion-a", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    salida = await preguntar(app, "sesion-b", "¿Y el último?")
    assert _llamadas(salida["messages"]) == []
    assert salida["messages"][-1].content.startswith("¿De qué cliente querés ver el último pedido?")
    assert len(salida["messages"]) == 2


async def test_el_historial_sobrevive_a_cerrar_y_reabrir_sqlite(ruta_db: str) -> None:
    async with abrir_agente(ruta_db) as app:
        await preguntar(app, "sesion-disco", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?")
    assert Path(ruta_db).stat().st_size > 0

    async with abrir_agente(ruta_db) as reabierto:
        estado = await reabierto.aget_state({"configurable": {"thread_id": "sesion-disco"}})
        assert len(estado.values["messages"]) == 6
        salida = await preguntar(reabierto, "sesion-disco", "¿Y el último?")
    assert "P-1023" in salida["messages"][-1].content


async def test_borrar_hilos_deja_el_thread_vacio(app: AppAgente) -> None:
    await preguntar(app, "sesion-borrar", "¿Cuántos pedidos tuvo el cliente 102?")
    await borrar_hilos(app, ["sesion-borrar"])
    estado = await app.aget_state({"configurable": {"thread_id": "sesion-borrar"}})
    assert estado.values == {}
