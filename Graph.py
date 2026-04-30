from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from Utility import (
    chat_worker, manager_node, worker_node,
    evaluator_node, route_from_chatbot,
    image_worker_node, route_from_manager
)
from State import Task_State
from memory import checkpointer
from Model import tools

tool_node = ToolNode(tools)

Graph = StateGraph(Task_State)

Graph.add_node("Chatbot",     chat_worker)
Graph.add_node("Manager",     manager_node)
Graph.add_node("ImageWorker", image_worker_node) 
Graph.add_node("Worker",      worker_node)
Graph.add_node("tools",       tool_node)
Graph.add_node("Evaluator",   evaluator_node)

Graph.add_edge(START, "Chatbot")
Graph.add_conditional_edges(
    "Chatbot", route_from_chatbot,
    {"Manager": "Manager", END: END}
)

# After Manager: split image vs research
Graph.add_conditional_edges(
    "Manager", route_from_manager,
    {"ImageWorker": "ImageWorker", "Worker": "Worker"}
)

# Image path: parallel models → straight to Evaluator
Graph.add_edge("ImageWorker", "Evaluator")

# Research path: LLM picks tools → tool execution → Evaluator
Graph.add_conditional_edges("Worker", tools_condition)
Graph.add_edge("tools",    "Evaluator")

Graph.add_edge("Evaluator", END)

Workflow = Graph.compile(checkpointer=checkpointer)
print(Workflow.get_graph().draw_ascii())