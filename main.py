"""Demo del agente: razonamiento multi-paso, memoria por thread_id y ciclo de retorno ante errores."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agente import RECURSION_LIMIT, RUTA_DB, abrir_agente, borrar_hilos, preguntar
from errors import AgenteError
from modelo import resolver_modelo
from traza import RUTA_TRAZA_JSON, RUTA_TRAZA_LOG, guardar_traza, registrar_turno

# (turno, thread_id, pregunta). Los tres primeros comparten thread_id: "¿Y el último?" y
# "¿Y Juan Pérez?" solo se entienden con el historial que guarda el checkpointer.
ESCENARIOS: list[tuple[str, str, str]] = [
    ("turno_1_multi_paso", "conversacion-demo-1", "¿Cuántos pedidos tuvo Ana García y cuál fue el total?"),
    ("turno_2_memoria", "conversacion-demo-1", "¿Y el último?"),
    ("turno_3_memoria_otro_cliente", "conversacion-demo-1", "¿Y Juan Pérez?"),
    ("prueba_reintento_nombre_incompleto", "conversacion-reintento-1", "¿Cuántos pedidos tiene García?"),
    ("prueba_error_y_aclaracion", "conversacion-error-1", "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?"),
    ("prueba_hilo_aislado", "conversacion-aislada-1", "¿Y el último?"),
]


async def main(
    ruta_db: str = RUTA_DB,
    ruta_json: Path = RUTA_TRAZA_JSON,
    ruta_log: Path = RUTA_TRAZA_LOG,
) -> None:
    turnos: dict[str, Any] = {}
    async with abrir_agente(ruta_db) as app:
        await borrar_hilos(app, {thread_id for _, thread_id, _ in ESCENARIOS})
        for turno, thread_id, pregunta in ESCENARIOS:
            salida = await preguntar(app, thread_id, pregunta, recursion_limit=RECURSION_LIMIT)
            turnos[turno] = registrar_turno(thread_id, pregunta, salida["messages"])
            print(f"\n## {turno} (thread_id={thread_id})")
            print("\n".join(turnos[turno]["traza_react"]))

        # Historial recuperado del checkpoint SQLite: los tres turnos del mismo thread_id.
        estado = await app.aget_state({"configurable": {"thread_id": "conversacion-demo-1"}})
        print(f"\n--- conversacion-demo-1: {len(estado.values['messages'])} mensajes en {ruta_db} ---")
        for mensaje in estado.values["messages"]:
            mensaje.pretty_print()

    provider, modelo = resolver_modelo()
    guardar_traza(
        {"llm": f"{provider}:{modelo}", "recursion_limit": RECURSION_LIMIT, "turnos": turnos},
        ruta_json,
        ruta_log,
    )
    print(f"\nTraza guardada en {ruta_json} y {ruta_log}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).with_name(".env"))
    try:
        asyncio.run(main())
    except AgenteError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)
