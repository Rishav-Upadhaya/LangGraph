from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# State Defined
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tools Defined
# These tools will be used by the agent to perform calculations
@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtract two numbers."""
    return a - b

@tool
def mul(a: int, b: int):
    """Multiply two numbers."""
    return a * b

tools = [add, subtract, mul] # List of tools available to the agent

# Model Defined
# Using Google Generative AI model for the agent
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")).bind_tools(tools)

# Nodes
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful assistant that can perform calculations using tools."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
# Graph Definition
# This graph defines the flow of the agent's actions
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

# Define a function to print the stream of messages
# This function will be used to display the agent's responses in a readable format

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Ram has 8 oranges and he ate 3 of it. Shyam said he will multiply the remaining oranges by 2. How many oranges do they have now? Also tell me how common this")]}
print_stream(app.stream(inputs, stream_mode="values"))
