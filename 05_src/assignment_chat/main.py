"""Main agent logic using LangGraph for the chat assistant."""

import os
from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv

from assignment_chat.prompts import SYSTEM_PROMPT
from assignment_chat.tools import ALL_TOOLS

# Load environment variables
load_dotenv(".env")
load_dotenv(".secrets")

# Initialize the chat model using course API gateway
chat_agent = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)


def call_model(state: MessagesState):
    """LLM decides whether to call a tool or respond directly."""
    response = chat_agent.bind_tools(ALL_TOOLS).invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def get_graph():
    """Build and compile the LangGraph agent."""
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    
    # Add edges
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    
    # Compile and return
    graph = builder.compile()
    return graph


# Create the graph instance
graph = get_graph()
