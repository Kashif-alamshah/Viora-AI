# Viora AI

Viora AI is a health-assistant agent that chats with users, analyzes uploaded medical images (skin and oral lesions), and pulls supporting research from PubMed and arXiv — all orchestrated as a LangGraph pipeline with a manager/evaluator pattern for routing and quality control.

## Features

- Conversational chatbot front-end with automatic conversation summarization for long sessions
- Image-based oncology screening for skin and oral lesions using parallel CNN/EfficientNet models
- Research-backed answers via PubMed and arXiv lookup tools
- Manager node that routes each query to the right path (image analysis vs. research/chat)
- Evaluator node that reviews results before responding
- Conversation memory persisted across turns (MongoDB-backed checkpointing)
- PHI/identity anonymization on relevant inputs

## Tech stack

- **Orchestration:** LangGraph, LangChain
- **LLMs:** OpenAI-compatible models (configurable per role: manager, worker, evaluator), Google Gemini support
- **ML models:** PyTorch (CNN / EfficientNet) for image classification
- **Memory:** MongoDB checkpointing
- **Anonymization:** Microsoft Presidio
- **NLP:** spaCy

## Project structure

```
Viora-AI/
├── Graph.py              # LangGraph wiring — nodes and edges
├── State.py               # Shared graph state schema
├── Model.py                # LLM clients (Manager, Worker, Evaluator) + tool bindings
├── Prompt.py                # Prompt templates for each node
├── Utility.py                 # Node logic (chatbot, manager, image worker, evaluator)
├── tools.py                    # PubMed/arXiv lookup + model tools
├── predictor.py / cnn_predictor.py        # Skin lesion models
├── oral_predictor.py / oral_efficientnet.py  # Oral lesion models
├── nn_models.py                  # Shared model definitions
├── anonymizer.py                   # PHI scrubbing
├── memory.py                        # Conversation checkpointing
├── Doctor/                           # (see source for details)
├── main.py                            # Entry point
└── requirements.txt
```

## Getting started

### Prerequisites
- Python 3.10+
- A MongoDB instance (for conversation memory)
- API keys for your chosen LLM provider(s)

### Installation

```bash
git clone https://github.com/Kashif-alamshah/Viora-AI.git
cd Viora-AI
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with at least:

```env
OPENAI_API_KEY=your_key_here
BASE_URL=https://api.openai.com/v1
MANAGER_MODEL=gpt-4o-mini
WORKER_MODEL=gpt-4o
EVALUATOR_MODEL=gpt-4o-mini
MONGODB_URI=your_mongodb_connection_string
```

Adjust model names and provider URL to match whichever OpenAI-compatible service you're using.

### Running

```bash
python main.py
```

Refer to `main.py` for the exact invocation pattern (CLI, API server, etc.) and adjust as needed for your use case.

## How it works (brief)

A message first hits a chatbot node, which handles casual conversation directly. If the message includes an image or looks like a research question, it's routed to a manager node, which classifies the task and sends it either to an image-analysis path (running the relevant lesion-detection models in parallel) or a research path (an LLM that can call PubMed/arXiv/model tools as needed). Both paths converge on an evaluator node that reviews the result before it's returned to the user.

## Disclaimer

Viora AI is a research/educational project and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
