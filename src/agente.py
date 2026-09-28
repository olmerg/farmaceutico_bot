"""Agente Farmacéutico con LangChain + NVIDIA Build API."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE
from langgraph.checkpoint.memory import MemorySaver

from tools import TOOLS

load_dotenv()

if not os.environ.get("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY no configurada. Copia .env-example a .env y agrega tu clave.")

MODELO = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")


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
    _registrar_perfil_modelo()
    llm = ChatNVIDIA(
        model=MODELO,
        temperature=1.0,
        top_p=0.95,
        max_completion_tokens=1024,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
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
