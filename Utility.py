import json
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END
from langchain_core.runnables import RunnableParallel, RunnableLambda
from predictor import predict
from cnn_predictor import cnn_predict
from Prompt import ManagerPrompt, EvaluatorPrompt, ChatbotPrompt
from Model import Manager, Worker_with_tools, Evaluator, Worker
from State import Task_State
from oral_predictor import oral_predict
from oral_efficientnet import oral_efficientnet_predict

# ── Routing helpers ───────────────────────────────────────────────

def route_from_chatbot(state: Task_State) -> str:
    return "Manager" if state["output"] == "ROUTE_TO_PIPELINE" else END


def route_from_manager(state: Task_State) -> str:
    return "ImageWorker" if state.get("task") == "image" else "Worker"


# ── Nodes ─────────────────────────────────────────────────────────

def chat_worker(state: Task_State) -> dict:
    messages = state.get("whole_messages", [])
    current_summary = state.get("summary", "")
    delete_messages = []
    
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content.lower()
            break

    if any(ext in last_human for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
        return {**state, "output": "ROUTE_TO_PIPELINE"}

    if any(word in last_human for word in ["research", "paper", "study", "pubmed", "arxiv"]):
        return {**state, "output": "ROUTE_TO_PIPELINE"}

    # ── SUMMARIZATION LOGIC ──
    MAX_MESSAGES = 10
    if len(messages) > MAX_MESSAGES:
        # Keep the last 4 messages for immediate back-and-forth context
        messages_to_summarize = messages[:-4]
        recent_messages = messages[-4:]
        
        # Format the old conversation to send to the summarizer
        convo_text = ""
        for m in messages_to_summarize:
            role = "Human" if isinstance(m, HumanMessage) else "AI"
            convo_text += f"{role}: {m.content}\n"
            if getattr(m, "id", None): 
                delete_messages.append(RemoveMessage(id=m.id))
        
        # Instruct the Worker model to compress the context
        summary_instruction = (
            f"Distill the following conversation into a concise summary. "
            f"Include any important medical context, symptoms, or past actions. "
            f"Combine this with the existing summary:\n\n"
            f"Existing Summary: {current_summary}\n\n"
            f"New Conversation to summarize:\n{convo_text}"
        )
        
        summary_response = Worker.invoke([SystemMessage(content=summary_instruction)])
        new_summary = summary_response.content
    else:
        new_summary = current_summary
        recent_messages = messages
        
    # Generate the actual chatbot response, providing the new summary and recent messages
    chain = ChatbotPrompt | Worker
    response = chain.invoke({
        "summary": new_summary, 
        "whole_messages": recent_messages
    })
    
    return {
        **state,
        "output": response.content,
        "summary": new_summary,  # Overwrite state with the latest summary
        "whole_messages": delete_messages + [response], 
    }




def manager_node(state: Task_State) -> dict:
    # Extract only the last human message
    last_human = ""
    for msg in reversed(state["whole_messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    chain = ManagerPrompt | Manager
    response = chain.invoke({"whole_messages": last_human})  # only last message
    parsed = json.loads(response.content)

    if any(word in last_human.lower() for word in ["oral", "mouth", "tongue", "gum", "lip"]):
        image_type = "oral"
    else:
        image_type = "skin"

    print(f"DEBUG manager image_type: {image_type}")
    print(f"DEBUG manager task: {parsed['task']}")

    # --- THE FIX: Wipe the internal scratchpad from previous turns ---
    # This deletes all old tool calls and image paths so the Worker doesn't get confused
    delete_messages = [RemoveMessage(id=m.id) for m in state.get("messages", []) if getattr(m, "id", None)]

    return {
        **state,
        "instruction": parsed["instruction"],
        "query":       parsed["query"],
        "task":        parsed["task"],
        "image_type":  image_type,
        
        # Add the delete requests BEFORE adding the new query
        "messages":    delete_messages + [HumanMessage(content=parsed["query"])],
    }


def image_worker_node(state: Task_State) -> dict:
    image_path = state["query"]
    image_type = state.get("image_type", "skin")
    print(f"DEBUG image_worker image_type: {image_type}")

    if image_type == "oral":
        parallel = RunnableParallel(
            oral_cnn=RunnableLambda(lambda p: oral_predict(p)),
            oral_efficientnet=RunnableLambda(lambda p: oral_efficientnet_predict(p)),
        )
        results = parallel.invoke(image_path)
        combined = (
            f"Oral CNN Result:\n{results['oral_cnn']}\n\n"
            f"Oral EfficientNet Result:\n{results['oral_efficientnet']}"
        )
    else:
        parallel = RunnableParallel(
            efficientnet=RunnableLambda(lambda p: predict(p)),
            cnn=RunnableLambda(lambda p: cnn_predict(p)),
        )
        results = parallel.invoke(image_path)
        combined = (
            f"EfficientNet Result:\n{results['efficientnet']}\n\n"
            f"CNN Result:\n{results['cnn']}"
        )

    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=combined)],
    }

def worker_node(state: Task_State) -> dict:
    """Research path only — image queries never reach this node."""
    worker_result = Worker_with_tools.invoke({"messages": state["messages"]})
    return {
        **state,
        "messages": state["messages"] + [worker_result],
    }


def evaluator_node(state: Task_State) -> dict:
    chain = EvaluatorPrompt | Evaluator
    evaluator_result = chain.invoke({"messages": state["messages"]})
    return {
        **state,
        "messages": state["messages"] + [evaluator_result],
        "output":   evaluator_result.content,
    }