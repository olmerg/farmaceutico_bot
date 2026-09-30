"""Agente Farmacéutico con LangChain + NVIDIA Build API."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from tools import TOOLS

load_dotenv()

PROVEEDOR = os.environ.get("PROVEEDOR", "nvidia").lower()

if PROVEEDOR == "nvidia":
    if not os.environ.get("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY no configurada. Copia .env-example a .env y agrega tu clave.")
    MODELO = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
elif PROVEEDOR == "groq":
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY no configurada. Agrega tu clave de Groq al .env.")
    MODELO = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    raise ValueError(f"Proveedor no soportado: {PROVEEDOR}. Usa 'nvidia' o 'groq'.")


def _registrar_perfil_modelo() -> None:
    """Registra el modelo en el catalogo estatico para evitar warnings."""
    MODEL_TABLE[MODELO] = Model(
        id=MODELO,
        model_type="chat",
        client="ChatNVIDIA",
        supports_tools=True,
        supports_structured_output=True,
        supports_thinking=True,
        thinking_param_enable={"chat_template_kwargs": {"enable_thinking": True}},
        thinking_param_disable={"chat_template_kwargs": {"enable_thinking": False}},
    )


def crear_agente():
    """Crea el agente farmacéutico con tool-calling."""
    if PROVEEDOR == "nvidia":
        _registrar_perfil_modelo()
        llm = ChatNVIDIA(
            model=MODELO,
            temperature=0,
            max_completion_tokens=1024,
            model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            model=MODELO,
            temperature=0,
            max_tokens=1024,
        )

    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=(
            "Eres un farmacéutico experto del INVIMA Colombia. "
            "Usa las herramientas para consultar el dataset. Responde en español."
        ),
        checkpointer=MemorySaver(),
        debug=False,
    )


def responder(agente, pregunta: str, thread_id: str = "sesion-1") -> str:
    """Envía una pregunta al agente y retorna la respuesta."""
    config = {"configurable": {"thread_id": thread_id}}
    resultado = agente.invoke({"messages": [("user", pregunta)]}, config=config)
    return resultado["messages"][-1].content


if __name__ == "__main__":
    agente = crear_agente()
    print(f"Agente listo. Modelo: {MODELO}")

    while True:
        pregunta = input("\n[Usuario]: ")
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        print(f"[Agente]: {responder(agente, pregunta)}")
