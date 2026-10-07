"""Chequeo offline: artefactos de la consigna, traza real incluida y demo con el doble del LLM (sin API)."""

import asyncio
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import agente
import main
from tests.modelo_local import ModeloPedidosLocal

RAIZ = Path(__file__).resolve().parent

CADENAS: dict[str, list[str]] = {
    "agente.py": [
        "grafo = StateGraph(MessagesState)",
        "async def nodo_modelo(state: MessagesState, config: RunnableConfig)",
        "await obtener_llm_con_herramientas().ainvoke(mensajes)",
        'grafo.add_node("modelo", nodo_modelo)',
        'grafo.add_node("herramientas", ToolNode(herramientas, handle_tool_errors=True))',
        'grafo.add_edge(START, "modelo")',
        "tools_condition",
        '{"tools": "herramientas", END: END}',
        'grafo.add_edge("herramientas", "modelo")',
        "llm_con_herramientas = llm.bind_tools(herramientas)",
        "AsyncSqliteSaver.from_conn_string(ruta_db)",
        "grafo.compile(checkpointer=checkpointer)",
        "RECURSION_LIMIT = 10",
        '"recursion_limit": recursion_limit',
        '"thread_id": thread_id',
        "except GraphRecursionError",
        "trim_messages",
    ],
    "herramientas.py": [
        "@tool(args_schema=BuscarClienteInput)",
        "def buscar_cliente_por_nombre(nombre: str) -> str:",
        "@tool(args_schema=ClienteIdInput)",
        "def buscar_pedidos(cliente_id: int) -> str:",
        "def buscar_ultimo_pedido(cliente_id: int) -> str:",
        "CLIENTES_DB",
        "PEDIDOS_DB",
        "herramientas = [buscar_cliente_por_nombre, buscar_pedidos, buscar_ultimo_pedido]",
    ],
    "modelo.py": ["ChatOpenAI(", 'provider == "anthropic"', "os.getenv(variable)", "401/key: falta"],
    "main.py": [
        "async with abrir_agente(ruta_db) as app:",
        "recursion_limit=RECURSION_LIMIT",
        'app.aget_state({"configurable": {"thread_id": "conversacion-demo-1"}})',
        "mensaje.pretty_print()",
        "asyncio.run(main())",
        "load_dotenv(",
    ],
}


def _ok(nombre: str, condicion: bool, detalle: str = "") -> None:
    print(f"[{'OK' if condicion else 'FALLO'}] {nombre}{f' - {detalle}' if detalle else ''}")
    if not condicion:
        raise SystemExit(1)


def chequear_repo() -> None:
    _ok("Python 3.12+", sys.version_info >= (3, 12), sys.version.split()[0])
    for archivo, cadenas in CADENAS.items():
        fuente = (RAIZ / archivo).read_text(encoding="utf-8")
        for cadena in cadenas:
            _ok(f"{archivo}: {cadena}", cadena in fuente)
    _ok("un solo StateGraph", sum((RAIZ / f).read_text(encoding="utf-8").count("StateGraph(") for f in CADENAS) == 1)
    for archivo in [*CADENAS, "historial.py", ".env.example"]:
        _ok(f"{archivo} sin claves", "sk-" not in (RAIZ / archivo).read_text(encoding="utf-8"))
    vacias = [l for l in (RAIZ / ".env.example").read_text(encoding="utf-8").splitlines() if l.endswith("_API_KEY=")]
    _ok(".env.example con las claves vacías", len(vacias) == 2)
    ignorados = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    _ok(".gitignore excluye .env y la base SQLite", ".env" in ignorados and "*.sqlite" in ignorados)


def chequear_traza_real() -> None:
    traza = json.loads((RAIZ / "traza_ejecucion.json").read_text(encoding="utf-8"))
    turnos = traza["turnos"]
    _ok("traza_ejecucion.json con el LLM real", traza["llm"].startswith("openai:"), traza["llm"])
    _ok("traza: recursion_limit 10", traza["recursion_limit"] == 10)
    llamadas = turnos["turno_1_multi_paso"]["herramientas_invocadas"]
    _ok("traza: turno 1 llama 2 herramientas", len(llamadas) >= 2, ", ".join(llamadas))
    _ok("traza: turno 2 usa el cliente del turno 1", turnos["turno_2_memoria"]["herramientas_invocadas"] == ["buscar_ultimo_pedido(cliente_id=102)"])
    reintento = turnos["prueba_reintento_nombre_incompleto"]["herramientas_invocadas"]
    _ok("traza: segundo intento tras ERROR", reintento[:2] == ["buscar_cliente_por_nombre(nombre='García')", "buscar_cliente_por_nombre(nombre='Ana García')"])
    _ok("traza: hilo aislado no llama herramientas", turnos["prueba_hilo_aislado"]["herramientas_invocadas"] == [])
    _ok("traza_ejecucion.log", (RAIZ / "traza_ejecucion.log").read_text(encoding="utf-8").startswith("## turno_1_multi_paso"))


def chequear_demo_offline() -> None:
    agente.obtener_llm_con_herramientas = lambda: ModeloPedidosLocal()  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as carpeta:
        tmp = Path(carpeta)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            asyncio.run(main.main(str(tmp / "checkpoints.sqlite"), tmp / "traza.json", tmp / "traza.log"))
        salida = buffer.getvalue()
        print(salida, end="")
        turnos = json.loads((tmp / "traza.json").read_text(encoding="utf-8"))["turnos"]
    _ok("demo: multi-paso", len(turnos["turno_1_multi_paso"]["herramientas_invocadas"]) == 2)
    _ok("demo: memoria por thread_id", "P-1023" in turnos["turno_2_memoria"]["respuesta"])
    _ok("demo: segundo intento", len(turnos["prueba_reintento_nombre_incompleto"]["herramientas_invocadas"]) == 3)
    _ok("demo: aclaración sin inventar", "confirmar" in turnos["prueba_error_y_aclaracion"]["respuesta"])
    _ok("demo: historial recuperado", "--- conversacion-demo-1: 16 mensajes en" in salida)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    chequear_repo()
    chequear_traza_real()
    chequear_demo_offline()
    print("[OK] validacion")
