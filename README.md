# Viora AI — LangGraph Architecture

Viora AI is a health-assistant agent built with LangGraph. It runs a `Chatbot` entry node for casual conversation, and only enters the full pipeline (`Manager → Worker/ImageWorker → Evaluator`) when the message contains an image or a research-style query. This document reflects the actual implementation in [Kashif-alamshah/Viora-AI](https://github.com/Kashif-alamshah/Viora-AI).

## Graph topology (`Graph.py`)

```
START
  │
  ▼
Chatbot ──(route_from_chatbot)──► END                [no pipeline needed]
  │
  ▼ (ROUTE_TO_PIPELINE)
Manager ──(route_from_manager)──┬──► ImageWorker ──► Evaluator ──► END
                                  └──► Worker ──(tools_condition)──► tools ──► Evaluator ──► END
```

Built with `StateGraph(Task_State)` and compiled with a `checkpointer` for cross-turn memory:

```python
Graph.add_node("Chatbot", chat_worker)
Graph.add_node("Manager", manager_node)
Graph.add_node("ImageWorker", image_worker_node)
Graph.add_node("Worker", worker_node)
Graph.add_node("tools", tool_node)
Graph.add_node("Evaluator", evaluator_node)

Graph.add_edge(START, "Chatbot")
Graph.add_conditional_edges("Chatbot", route_from_chatbot, {"Manager": "Manager", END: END})
Graph.add_conditional_edges("Manager", route_from_manager, {"ImageWorker": "ImageWorker", "Worker": "Worker"})
Graph.add_edge("ImageWorker", "Evaluator")
Graph.add_conditional_edges("Worker", tools_condition)
Graph.add_edge("tools", "Evaluator")
Graph.add_edge("Evaluator", END)
```

## State (`State.py`)

```python
class Task_State(TypedDict):
    instruction: str
    query: str
    task: str
    image_type: str
    output: str
    summary: str                              # running conversation summary
    messages: Annotated[list, add_messages]           # scratchpad (tool calls, image results)
    whole_messages: Annotated[list[BaseMessage], add_messages]  # full chat history
```

## Nodes (`Utility.py`)

### Chatbot (`chat_worker`)
The entry point for every message. It:
- Checks the last `HumanMessage` for an image file extension (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) or research keywords (`research`, `paper`, `study`, `pubmed`, `arxiv`). Either match sets `output = "ROUTE_TO_PIPELINE"`.
- If the conversation exceeds 10 messages, it summarizes the older messages via the `Worker` LLM and keeps only the last 4 for immediate context, deleting the rest with `RemoveMessage`.
- Otherwise responds directly as a conversational chatbot using `ChatbotPrompt | Worker`, ending the turn without entering the pipeline.

`route_from_chatbot` sends the state to `Manager` only when `output == "ROUTE_TO_PIPELINE"`, otherwise the graph ends.

### Manager (`manager_node`)
Runs `ManagerPrompt | Manager` (an LLM call) on the last human message, parsing a JSON response into `instruction`, `query`, and `task`. It also infers `image_type` ("oral" vs "skin") by keyword-matching ("oral", "mouth", "tongue", "gum", "lip"). It wipes the prior turn's scratchpad (`messages`) with `RemoveMessage` so the worker doesn't see stale tool calls or image paths from earlier turns.

`route_from_manager` sends `task == "image"` to `ImageWorker`, everything else to `Worker`.

### ImageWorker (`image_worker_node`)
This is the worker layer with direct ML model access. It branches on `image_type`:

| image_type | Models run in parallel (`RunnableParallel`) | Source file |
|---|---|---|
| `oral` | `oral_predict()` + `oral_efficientnet_predict()` | `oral_predictor.py`, `oral_efficientnet.py` |
| `skin` (default) | `predict()` (EfficientNet) + `cnn_predict()` (CNN) | `predictor.py`, `cnn_predictor.py` |

Both models in a pair run concurrently on the same image path; their results are combined into a single message and appended to `messages` before going straight to `Evaluator` — no tool-calling LLM in this path.

### Worker (`worker_node`) — research path only
Invoked when `task != "image"`. Calls `Worker_with_tools` (`WorkerPrompt | Worker.bind_tools(tools, tool_choice="any", parallel_tool_calls=True)`), which can call any of:

| Tool | Purpose | Source |
|---|---|---|
| `get_top_pubmed_papers` | PubMed literature search | `tools.py` |
| `get_top_arxiv_papers` | arXiv paper search | `tools.py` |
| `skin_cancer_predictor` | EfficientNet skin model, exposed as a tool | `tools.py` |
| `skin_cancer_cnn_predictor` | CNN skin model, exposed as a tool | `tools.py` |

`tools_condition` (from `langgraph.prebuilt`) checks whether the LLM requested a tool call; if so the graph executes the matching tool node, then proceeds to `Evaluator`.

### Evaluator (`evaluator_node`)
Runs `EvaluatorPrompt | Evaluator` over the full `messages` history (model predictions, tool results, etc.), appends its judgment to `messages`, and writes the final text to `state["output"]`. This is the manager-evaluator pattern: a separate LLM call reviews everything the worker/image path produced before the graph ends.

## Models (`Model.py`)
Three separate `ChatOpenAI` instances, each configurable via env vars (`MANAGER_MODEL`, `EVALUATOR_MODEL`, `WORKER_MODEL`) against a shared `BASE_URL`:

```python
Manager   = ChatOpenAI(model=os.getenv("MANAGER_MODEL"), ...)
Evaluator = ChatOpenAI(model=os.getenv("EVALUATOR_MODEL"), ...)
Worker    = ChatOpenAI(model=os.getenv("WORKER_MODEL"), ...)
```

This lets you run a cheaper/faster model for routing (`Manager`) and a stronger model for evaluation, independently.

## Supporting modules
- **`memory.py`** — `checkpointer`, persists `Task_State` across turns (likely per `thread_id`) so the graph can be invoked conversationally.
- **`anonymizer.py`** — scrubs PHI/identifying information, presumably applied to inputs or outputs given the health-data context.
- **`nn_models.py`** — shared neural-net model loading/definitions used by the predictor files.
- **`Doctor/`** — separate module, not yet inspected (worth documenting once reviewed).

## Notes on the manager-evaluator pattern as implemented
- There is **one Manager** (routes to image vs research) and **one Evaluator** (reviews final output) — both LLM calls, not a model-per-worker registry.
- The "workers" with direct ML model access are the four oncology/lesion classifiers inside `ImageWorker`, called directly as Python functions wrapped in `RunnableParallel`, not as LangChain tools.
- The two skin models (`skin_cancer_predictor`, `skin_cancer_cnn_predictor`) are *also* exposed as LangChain tools to the `Worker` node, so the LLM-driven research path can invoke them on demand in addition to the always-on parallel path in `ImageWorker`.
- There is currently **no retry edge** from `Evaluator` back to `Manager` — `Evaluator` always proceeds to `END`. If you want the manager-evaluator retry loop described in your earlier sketch, that conditional edge would need to be added.

## Suggested next steps
- [ ] Add a conditional edge from `Evaluator` back to `Manager` (or a dedicated retry node) gated on a confidence/quality check, to fully realize the retry loop.
- [ ] Document `Doctor/` and `nn_models.py` once reviewed.
- [ ] Confirm what `anonymizer.py` is applied to (raw upload, query text, or stored memory).
- [ ] Decide whether oral/skin model selection should also be exposed in the `Worker` tool-calling path (currently only skin models are tools).
