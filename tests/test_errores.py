import httpx2
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

import agente
from agente import AppAgente, preguntar
from errors import (
    AgenteError,
    ClaveAPIError,
    ConfiguracionError,
    ConsultaInvalidaError,
    CuotaError,
    RedError,
    SalidaTruncadaError,
)
from modelo import clasificar_error, crear_llm
from tests.modelo_local import ModeloConFallos

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
_LLM_REAL = agente.obtener_llm_con_herramientas


def _status(codigo: int, clase: type[openai.APIStatusError], texto: str) -> openai.APIStatusError:
    return clase(texto, response=httpx2.Response(codigo, request=_REQUEST), body=None)


def _con_fallos(monkeypatch: pytest.MonkeyPatch, doble: ModeloConFallos) -> ModeloConFallos:
    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: doble)
    return doble


# --- 401 / key ---------------------------------------------------------------


def test_falta_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ClaveAPIError, match=r"^401/key: falta OPENAI_API_KEY en las variables de entorno\."):
        crear_llm()


def test_falta_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ClaveAPIError, match="falta ANTHROPIC_API_KEY"):
        crear_llm("anthropic")


async def test_401_desde_el_grafo_no_reintenta(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    doble = _con_fallos(monkeypatch, ModeloConFallos(_status(401, openai.AuthenticationError, "Incorrect API key provided")))
    with pytest.raises(ClaveAPIError, match=r"^401/key: el proveedor rechazó la API key\. No se reintenta"):
        await preguntar(app, "e-401", "¿Cuántos pedidos tuvo el cliente 102?")
    assert doble.llamadas == 1


async def test_sin_clave_el_grafo_devuelve_401(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(agente, "obtener_llm_con_herramientas", lambda: crear_llm())
    with pytest.raises(ClaveAPIError, match="falta OPENAI_API_KEY"):
        await preguntar(app, "e-sin-clave", "¿Cuántos pedidos tuvo el cliente 102?")


# --- 429 / cuota -------------------------------------------------------------


async def test_429_reintenta_y_responde(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    doble = _con_fallos(monkeypatch, ModeloConFallos(_status(429, openai.RateLimitError, "Rate limit reached")))
    salida = await preguntar(app, "e-429", "hola")
    assert salida["messages"][-1].content == "Listo."
    assert doble.llamadas == 2


async def test_429_agotado(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    fallos = [_status(429, openai.RateLimitError, "Rate limit reached") for _ in range(3)]
    doble = _con_fallos(monkeypatch, ModeloConFallos(*fallos))
    with pytest.raises(CuotaError, match=r"^429/cuota: .*reintentos agotados: 3 intentos"):
        await preguntar(app, "e-429-agotado", "hola")
    assert doble.llamadas == 3


async def test_backoff_duplica_la_espera(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    esperas: list[float] = []

    async def dormir(segundos: float) -> None:
        esperas.append(segundos)

    monkeypatch.setattr(agente, "ESPERA_INICIAL_S", 0.5)
    monkeypatch.setattr("agente.asyncio.sleep", dormir)
    _con_fallos(monkeypatch, ModeloConFallos(*[_status(429, openai.RateLimitError, "Rate limit") for _ in range(3)]))
    with pytest.raises(CuotaError):
        await preguntar(app, "e-backoff", "hola")
    assert esperas == [0.5, 1.0]


# --- red / timeout -----------------------------------------------------------


async def test_timeout_reintenta_y_responde(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    doble = _con_fallos(monkeypatch, ModeloConFallos(openai.APITimeoutError(request=_REQUEST)))
    salida = await preguntar(app, "e-timeout", "hola")
    assert salida["messages"][-1].content == "Listo."
    assert doble.llamadas == 2


async def test_reintentos_agotados_usan_el_ultimo_fallo(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    _con_fallos(
        monkeypatch,
        ModeloConFallos(
            _status(429, openai.RateLimitError, "Rate limit"),
            _status(429, openai.RateLimitError, "Rate limit"),
            openai.APIConnectionError(request=_REQUEST),
        ),
    )
    with pytest.raises(RedError, match=r"^red/timeout: no se pudo hablar con el proveedor del LLM\. .*reintentos agotados"):
        await preguntar(app, "e-ultimo", "hola")


@pytest.mark.parametrize(
    ("excepcion", "familia"),
    [
        (TimeoutError("read timed out"), RedError),
        (ConnectionError("reset by peer"), RedError),
        (_status(503, openai.InternalServerError, "Service unavailable"), RedError),
        (_status(429, openai.RateLimitError, "You exceeded your current quota"), CuotaError),
        (_status(401, openai.AuthenticationError, "Incorrect API key"), ClaveAPIError),
        (ValueError("algo raro"), AgenteError),
    ],
)
def test_clasificar_error(excepcion: Exception, familia: type[AgenteError]) -> None:
    clasificado = clasificar_error(excepcion)
    assert type(clasificado) is familia


async def test_error_no_transitorio_no_reintenta(app: AppAgente, monkeypatch: pytest.MonkeyPatch) -> None:
    doble = _con_fallos(monkeypatch, ModeloConFallos(ValueError("algo raro")))
    with pytest.raises(AgenteError, match=r"^Error del LLM: ValueError: algo raro"):
        await preguntar(app, "e-otro", "hola")
    assert doble.llamadas == 1


# --- salida truncada ---------------------------------------------------------


@pytest.mark.parametrize("metadata", [{"finish_reason": "length"}, {"stop_reason": "max_tokens"}])
async def test_salida_truncada_no_se_guarda(app: AppAgente, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, str]) -> None:
    cortada = AIMessage(content="Ana García tuvo 3 ped", response_metadata=metadata)
    _con_fallos(monkeypatch, ModeloConFallos(respuesta=cortada))
    with pytest.raises(SalidaTruncadaError, match=r"^Salida truncada: el modelo cortó la respuesta por max_tokens"):
        await preguntar(app, "e-truncada", "hola")
    estado = await app.aget_state({"configurable": {"thread_id": "e-truncada"}})
    assert all(m.content != cortada.content for m in estado.values["messages"])


# --- validación Pydantic de la consulta --------------------------------------


@pytest.mark.parametrize(
    ("thread_id", "mensaje", "kwargs", "esperado"),
    [
        ("t1", "   ", {}, "mensaje: String should have at least 1 character"),
        ("", "hola", {}, "thread_id: String should have at least 1 character"),
        ("hilo con espacios", "hola", {}, "thread_id: String should match pattern"),
        ("t1", "hola", {"recursion_limit": 100}, "recursion_limit: Input should be less than or equal to 50"),
        ("t1", "hola", {"max_mensajes": 1}, "max_mensajes: Input should be greater than or equal to 2"),
    ],
)
async def test_consulta_invalida(
    app: AppAgente, thread_id: str, mensaje: str, kwargs: dict[str, int], esperado: str
) -> None:
    with pytest.raises(ConsultaInvalidaError, match=f"^Consulta inválida: {esperado}"):
        await preguntar(app, thread_id, mensaje, **kwargs)


# --- configuración -----------------------------------------------------------


def test_provider_invalido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ConfiguracionError, match=r"^LLM_PROVIDER inválido: 'gemini'\. Valores soportados: openai, anthropic\.$"):
        crear_llm()


def test_crea_chat_openai_con_los_parametros_de_la_llamada(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-test")
    llm = crear_llm("openai", modelo="gpt-4.1-mini", max_tokens=256)
    assert isinstance(llm, ChatOpenAI)
    assert (llm.model_name, llm.max_tokens, llm.max_retries, llm.temperature) == ("gpt-4.1-mini", 256, 0, 0.0)


def test_crea_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-test")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    llm = crear_llm(max_tokens=300)
    assert isinstance(llm, ChatAnthropic)
    assert (llm.model, llm.max_tokens) == ("claude-haiku-4-5", 300)


def test_bind_tools_vincula_las_tres_herramientas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    _LLM_REAL.cache_clear()
    try:
        vinculado = _LLM_REAL()
    finally:
        _LLM_REAL.cache_clear()
    nombres = [t["function"]["name"] for t in vinculado.kwargs["tools"]]  # type: ignore[attr-defined]
    assert nombres == ["buscar_cliente_por_nombre", "buscar_pedidos", "buscar_ultimo_pedido"]
