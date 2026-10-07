"""Creación del LLM (OpenAI o Anthropic) y clasificación de sus errores."""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from errors import (
    AgenteError,
    ClaveAPIError,
    ConfiguracionError,
    CuotaError,
    RedError,
)

MODELO_OPENAI = "gpt-4o-mini"
MODELO_ANTHROPIC = "claude-haiku-4-5"
MAX_TOKENS = 1024


def resolver_modelo(provider: str | None = None, modelo: str | None = None) -> tuple[str, str]:
    """Devuelve (provider, modelo) a partir de los argumentos o de LLM_PROVIDER / OPENAI_MODEL / ANTHROPIC_MODEL."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return provider, modelo or os.getenv("OPENAI_MODEL") or MODELO_OPENAI
    if provider == "anthropic":
        return provider, modelo or os.getenv("ANTHROPIC_MODEL") or MODELO_ANTHROPIC
    raise ConfiguracionError(
        f"LLM_PROVIDER inválido: '{provider}'. Valores soportados: openai, anthropic."
    )


def crear_llm(
    provider: str | None = None,
    modelo: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> BaseChatModel:
    """Crea el chat model del agente a partir de las variables de entorno.

    Args:
        provider: "openai" o "anthropic". Si no se pasa, sale de LLM_PROVIDER (default "openai").
        modelo: nombre del modelo. Si no se pasa, OPENAI_MODEL o ANTHROPIC_MODEL.
        temperature: 0 para que la elección de herramientas sea estable.
        max_tokens: tope de la respuesta. Una respuesta cortada se rechaza en el nodo modelo.

    Los reintentos del SDK quedan en 0: los hace el nodo modelo, con backoff y
    con el mensaje de la familia de error.
    """
    provider, modelo = resolver_modelo(provider, modelo)
    variable = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(variable):
        raise ClaveAPIError(
            f"401/key: falta {variable} en las variables de entorno. "
            "Copiá .env.example a .env y cargá la clave. No se reintenta."
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model_name=modelo,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
            max_retries=0,
            timeout=None,
            stop=None,
        )
    return ChatOpenAI(
        model=modelo,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        max_retries=0,
    )


def clasificar_error(exc: BaseException) -> AgenteError:
    """Traduce una excepción del SDK (OpenAI, Anthropic, httpx) a la familia de error del agente."""
    if isinstance(exc, AgenteError):
        return exc
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    detalle = f"{type(exc).__name__}: {exc}"
    texto = detalle.lower()
    if status == 401 or "authentication" in texto or "api key" in texto or "api_key" in texto:
        return ClaveAPIError(
            "401/key: el proveedor rechazó la API key. No se reintenta: "
            f"revisá la clave en .env. Detalle: {detalle}"
        )
    if status == 429 or "rate limit" in texto or "ratelimit" in texto or "quota" in texto:
        return CuotaError(
            f"429/cuota: el proveedor limitó las llamadas (rate limit o cuota). Detalle: {detalle}"
        )
    if (
        status in (408, 500, 502, 503, 504)
        or isinstance(exc, (TimeoutError, ConnectionError))
        or "timeout" in texto
        or "timed out" in texto
        or "connection" in texto
    ):
        return RedError(f"red/timeout: no se pudo hablar con el proveedor del LLM. Detalle: {detalle}")
    return AgenteError(f"Error del LLM: {detalle}")
