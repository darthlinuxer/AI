from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import tool  # Import the tool decorator
from langchain_core.messages import SystemMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(name="gpt-4o")

@tool
def add(a:int,b:int)->int:
    """Add two numbers together
    Args:
        a (int): First number
        b (int): Second number
    Returns:
        int: Sum of the two numbers
    """
    return a+b

tools = [add]
llm = llm.bind_tools(tools)

def chatbot(state: State)-> State:
    response = llm.invoke(state["messages"])
    state["messages"]= [response]
    return state

def route_tools(state: State):
    if messages:= state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    print(ai_message)
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "tools", 
        END: END},
)
graph = graph_builder.compile()
#graph = graph.with_config({"recursion_limit": 5})

initial_state = State(messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Add 2 and 5")
    ])

result = graph.invoke(initial_state)
[msg.pretty_print() for msg in result["messages"]]