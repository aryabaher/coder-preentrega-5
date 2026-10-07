"""Errores controlados del agente. Cada familia tiene su prefijo en el mensaje."""


class AgenteError(Exception):
    """Fallo del agente convertido en error de aplicación."""

    reintentable: bool = False


class ClaveAPIError(AgenteError):
    """401/key: falta la API key o el proveedor la rechazó. No se reintenta."""


class CuotaError(AgenteError):
    """429/cuota: rate limit o cuota del proveedor. Se reintenta con backoff."""

    reintentable = True


class RedError(AgenteError):
    """red/timeout: corte de red, timeout o 5xx del proveedor. Se reintenta con backoff."""

    reintentable = True


class SalidaTruncadaError(AgenteError):
    """El modelo cortó la respuesta por max_tokens. No se guarda a medias."""


class LimiteRecursionError(AgenteError):
    """El grafo llegó a recursion_limit sin una respuesta final."""


class ConsultaInvalidaError(AgenteError):
    """La consulta no pasa la validación Pydantic (mensaje vacío, thread_id, límites)."""


class ConfiguracionError(AgenteError):
    """LLM_PROVIDER u otra variable de entorno con un valor no soportado."""
