# Viora AI — LangGraph Architecture

Viora AI is a health-assistant agent built on LangGraph. It accepts text questions plus optional file uploads (PDF/image), routes the request through a manager agent to one or more specialist worker nodes backed by custom ML models, validates the result with an evaluator, and synthesizes a final response.

## High-level flow

```
User input (text + optional PDF/image)
        │
        ▼
Multimodal detector  ── parses file type, extracts text/visual content
        │
        ▼
Manager agent  ── classifies intent, decides which worker(s) to dispatch
        │
        ├──► Radiology worker     ──► Imaging oncology model
        ├──► Pathology worker     ──► Pathology NLP model
        ├──► Symptom worker       ──► Triage classifier model
        └──► General QA worker    ──► Health QA LLM / RAG
        │
        ▼
Evaluator node  ── checks confidence, flags hallucination risk
        │
        ├── pass  ──► Response synthesizer ──► Final response
        └── retry ──► back to Manager agent
```

## Components

### 1. Multimodal detector
Entry node. Detects whether the input includes a PDF or image, parses it (OCR / layout extraction for PDFs, basic preprocessing for images), and normalizes everything into a single state object before the manager sees it.

### 2. Manager agent
The orchestrator. Owns the `AgentState` and uses LangGraph `conditional_edges` to route to one or more worker nodes based on intent classification (e.g. "this is a radiology scan" → radiology worker; "this is a general question" → QA worker). Multiple workers can be fanned out in parallel via `asyncio.gather`.

### 3. Worker nodes
Each worker is a thin LangGraph node, not the model itself. Its job:
1. **Pre-process** — resize an image, extract PDF text, tokenize symptom text, etc.
2. **Call your ML model** — local `model.predict()`, a FastAPI inference endpoint, or a hosted endpoint (SageMaker, Vertex AI, etc.).
3. **Post-process** — package the prediction with a confidence score and write it back into state.

Current workers:
| Worker | Backing model |
|---|---|
| Radiology | Imaging oncology model |
| Pathology | Pathology NLP / report parsing model |
| Symptom | Triage classifier model |
| General QA | Health QA LLM / RAG |

All workers resolve their model from a **shared model registry**, so model versions can be swapped (`radiology_v2` → `radiology_v3`) without touching graph logic.

### 4. Evaluator node
Implements the manager-evaluator pattern. Reads confidence scores from whichever workers ran, checks for hallucination risk and low-confidence predictions, and either:
- **passes** the result downstream to synthesis, or
- **retries** by routing back to the manager (different model, stricter parameters, or escalation).

### 5. Response synthesizer
Merges outputs from all workers that contributed, attaches source/model citations, formats the final reply, and pulls conversation history from the state store.

### 6. Tool layer
Supporting tools available to the graph: medical RAG retrieval, web search, EHR/FHIR connectors, human clinician escalation, and audit logging.

### 7. State store
Holds conversation history and intermediate state across turns, accessible by the manager and synthesizer.

## Suggested implementation notes

- Use `StateGraph` with a typed `AgentState` (TypedDict or Pydantic model) covering: raw input, parsed file content, intent, worker outputs, confidence scores, retry count.
- Manager → workers: `conditional_edges` based on intent classification output.
- Worker → ML model: keep this call inside the worker node function; do not expose the model directly to the graph.
- Evaluator → `END` or back to manager: implement as a conditional edge checking a confidence threshold (e.g. `< 0.7` triggers retry, capped at N retries to avoid infinite loops).
- Parallel worker execution: use LangGraph's fan-out/fan-in pattern with `asyncio.gather` for workers running concurrently, then a join node before the evaluator.

## Next steps / open questions
- [ ] Define confidence threshold(s) per worker type
- [ ] Define max retry count before forced human escalation
- [ ] Decide model serving approach (local vs hosted endpoint) per worker
- [ ] Finalize state store schema
