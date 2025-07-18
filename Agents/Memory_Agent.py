from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] 

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

def process(state: AgentState) -> AgentState:
    """Process the state by invoking the LLM with the messages."""

    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))  # Append AI response to messages
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    # print(result["messages"])
    conversation_history = result["messages"]  # Update conversation history with the latest messages
    user_input = input("Enter: ")

with open("conversation_history.text", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n") 
    f.write("End of Conversation")  # Add a newline at the end of the file

print("Conversation history saved to conversation_history.text")