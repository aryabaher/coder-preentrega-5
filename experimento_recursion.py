"""Experimento: la pregunta de "García" (3 herramientas, 7 pasos) con distintos recursion_limit.

Uso:
    python experimento_recursion.py
    python experimento_recursion.py --limites 5 10
"""

from requisitos import exigir_python_312

exigir_python_312()

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from agente import RECURSION_LIMIT, RUTA_DB, abrir_agente, borrar_hilos, preguntar
from errors import AgenteError, LimiteRecursionError
from traza import lineas_react, mensajes_del_turno

PREGUNTA = "¿Cuántos pedidos tiene García?"
LIMITES = (5, 7, 10)


async def probar_limites(limites: Sequence[int] = LIMITES, ruta_db: str = RUTA_DB) -> dict[int, str]:
    """Corre la pregunta de "García" con cada recursion_limit, en un thread_id nuevo por límite.

    Si el límite corta el ciclo, imprime hasta dónde llegó el agente y el error controlado,
    y repite la pregunta en el mismo thread con RECURSION_LIMIT para mostrar que el hilo
    sigue usable. Devuelve {limite: "cortó" | "respondió"}.
    """
    resultados: dict[int, str] = {}
    async with abrir_agente(ruta_db) as app:
        for limite in limites:
            thread_id = f"experimento-recursion-{limite}"
            await borrar_hilos(app, [thread_id])
            print(f"\n## recursion_limit={limite} (thread_id={thread_id})")
            try:
                salida = await preguntar(app, thread_id, PREGUNTA, recursion_limit=limite)
            except LimiteRecursionError as exc:
                resultados[limite] = "cortó"
                estado = await app.aget_state({"configurable": {"thread_id": thread_id}})
                print("\n".join(lineas_react(estado.values["messages"])))
                print(f"Error controlado: {exc}")
                print(f"\n# Mismo thread_id, de nuevo con recursion_limit={RECURSION_LIMIT}")
                salida = await preguntar(app, thread_id, PREGUNTA, recursion_limit=RECURSION_LIMIT)
            else:
                resultados[limite] = "respondió"
            print("\n".join(lineas_react(mensajes_del_turno(salida["messages"]))))
    return resultados


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description="El caso García con distintos recursion_limit.")
    parser.add_argument("--limites", type=int, nargs="+", default=list(LIMITES))
    parser.add_argument("--db", default=RUTA_DB)
    args = parser.parse_args()
    try:
        resultados = asyncio.run(probar_limites(args.limites, args.db))
    except AgenteError as exc:
        print(f"Error controlado: {exc}")
        raise SystemExit(1)
    print("\nResumen: " + ", ".join(f"recursion_limit={n} {r}" for n, r in resultados.items()))
