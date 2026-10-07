import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from tests.modelo_local import llamada
from traza import guardar_traza, herramientas_invocadas, lineas_react, mensajes_del_turno, registrar_turno, serializar_traza


def _turno() -> list[BaseMessage]:
    pedir_id = llamada("buscar_cliente_por_nombre", nombre="Ana García")
    pedir_pedidos = llamada("buscar_pedidos", cliente_id=102)
    return [
        HumanMessage(content="viejo"),
        AIMessage(content="respuesta vieja"),
        HumanMessage(content="¿Cuántos pedidos tuvo Ana García?"),
        pedir_id,
        ToolMessage(content='{"nombre": "Ana García", "cliente_id": 102}', name="buscar_cliente_por_nombre", tool_call_id=pedir_id.tool_calls[0]["id"]),
        pedir_pedidos,
        ToolMessage(content='{"cliente_id": 102, "pedidos": 3, "total": 14500}', name="buscar_pedidos", tool_call_id=pedir_pedidos.tool_calls[0]["id"]),
        AIMessage(content="Ana García tuvo 3 pedidos por un total de $14.500."),
    ]


def test_mensajes_del_turno_arranca_en_la_ultima_pregunta() -> None:
    assert mensajes_del_turno(_turno())[0].content == "¿Cuántos pedidos tuvo Ana García?"


def test_serializar_traza_marca_llamadas_y_observaciones() -> None:
    traza = serializar_traza(mensajes_del_turno(_turno()))
    assert [t["tipo"] for t in traza] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert traza[1]["tool_calls"] == [{"nombre": "buscar_cliente_por_nombre", "argumentos": {"nombre": "Ana García"}}]
    assert traza[4]["herramienta"] == "buscar_pedidos" and traza[4]["status"] == "success"


def test_herramientas_invocadas_en_orden() -> None:
    assert herramientas_invocadas(_turno()) == [
        "buscar_cliente_por_nombre(nombre='Ana García')",
        "buscar_pedidos(cliente_id=102)",
    ]


def test_lineas_react_con_el_formato_del_ciclo() -> None:
    assert lineas_react(mensajes_del_turno(_turno())) == [
        'Usuario: "¿Cuántos pedidos tuvo Ana García?"',
        "→ El agente decide usar la herramienta: buscar_cliente_por_nombre(nombre='Ana García')",
        '→ La herramienta devuelve: {"nombre": "Ana García", "cliente_id": 102}',
        "→ El agente razona: todavía le falta un dato → llama otra herramienta.",
        "→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)",
        '→ La herramienta devuelve: {"cliente_id": 102, "pedidos": 3, "total": 14500}',
        "→ El agente razona: ya tiene los datos → responde.",
        'Respuesta: "Ana García tuvo 3 pedidos por un total de $14.500."',
    ]


def test_lineas_react_marca_el_segundo_intento_y_la_aclaracion() -> None:
    primero = llamada("buscar_cliente_por_nombre", nombre="García")
    segundo = llamada("buscar_cliente_por_nombre", nombre="Roberto")
    mensajes: list[BaseMessage] = [
        HumanMessage(content="¿Pedidos de García?"),
        primero,
        ToolMessage(content="ERROR: 'García' es un nombre incompleto.", name="buscar_cliente_por_nombre", tool_call_id=primero.tool_calls[0]["id"]),
        segundo,
        ToolMessage(content="ERROR: no se encontró ningún cliente", name="buscar_cliente_por_nombre", tool_call_id=segundo.tool_calls[0]["id"]),
        AIMessage(content="¿Me confirmás el nombre?"),
    ]
    lineas = lineas_react(mensajes)
    assert "→ El agente razona: la herramienta devolvió ERROR → hace un segundo intento." in lineas
    assert lineas[-2] == "→ El agente razona: la herramienta devolvió ERROR → pide una aclaración, no inventa."
    assert lineas_react([HumanMessage(content="¿Y el último?"), AIMessage(content="¿De qué cliente?")])[1] == (
        "→ El agente razona: no necesita herramientas o le falta contexto → responde."
    )


def test_guardar_traza_escribe_json_y_log(tmp_path: Path) -> None:
    turno = registrar_turno("conversacion-demo-1", "¿Cuántos pedidos tuvo Ana García?", _turno())
    assert turno["mensajes_en_checkpoint"] == 8
    assert len(turno["herramientas_invocadas"]) == 2
    ruta_json, ruta_log = tmp_path / "traza.json", tmp_path / "traza.log"
    guardar_traza({"llm": "openai:gpt-4o-mini", "recursion_limit": 10, "turnos": {"turno_1_multi_paso": turno}}, ruta_json, ruta_log)
    leido = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert leido["turnos"]["turno_1_multi_paso"]["respuesta"] == "Ana García tuvo 3 pedidos por un total de $14.500."
    log = ruta_log.read_text(encoding="utf-8")
    assert log.startswith("## turno_1_multi_paso (thread_id=conversacion-demo-1)\n")
    assert "→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)" in log
