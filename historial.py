"""Recupera un thread_id de checkpoints.sqlite en un proceso nuevo y, opcionalmente, sigue la charla.

Uso:
    python historial.py conversacion-demo-1
    python historial.py conversacion-demo-1 "¿En qué estado está ese pedido?"
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from agente import RECURSION_LIMIT, RUTA_DB, abrir_agente, preguntar
from errors import AgenteError
from traza import extraer_texto, herramientas_invocadas, mensajes_del_turno


async def mostrar_historial(thread_id: str, pregunta: str | None = None, ruta_db: str = RUTA_DB) -> int:
    """Imprime el historial guardado del thread y, si hay pregunta, la manda con ese contexto.

    Devuelve la cantidad de mensajes recuperados del checkpoint (0 si el thread no existe).
    """
    async with abrir_agente(ruta_db) as app:
        estado = await app.aget_state({"configurable": {"thread_id": thread_id}})
        mensajes = estado.values.get("messages", [])
        print(f"--- {thread_id}: {len(mensajes)} mensajes recuperados de {ruta_db} ---")
        for mensaje in mensajes:
            mensaje.pretty_print()
        if pregunta:
            salida = await preguntar(app, thread_id, pregunta, recursion_limit=RECURSION_LIMIT)
            turno = mensajes_del_turno(salida["messages"])
            print(f"\nUsuario: {pregunta}")
            print(f"Herramientas: {herramientas_invocadas(turno)}")
            print(f"Respuesta: {extraer_texto(turno[-1])}")
        return len(mensajes)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description="Historial de un thread_id guardado en SQLite.")
    parser.add_argument("thread_id")
    parser.add_argument("pregunta", nargs="?")
    parser.add_argument("--db", default=RUTA_DB)
    args = parser.parse_args()
    try:
        asyncio.run(mostrar_historial(args.thread_id, args.pregunta, args.db))
    except AgenteError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)
