from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools import get_top_pubmed_papers, get_top_arxiv_papers, skin_cancer_cnn_predictor, skin_cancer_predictor
from Prompt import ManagerPrompt, WorkerPrompt

import os

load_dotenv()


Manager = ChatOpenAI(
    model=os.getenv("MANAGER_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)



Evaluator = ChatOpenAI(
    model=os.getenv("EVALUATOR_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

Worker = ChatOpenAI(
    model=os.getenv("WORKER_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

tools = [get_top_pubmed_papers, get_top_arxiv_papers, skin_cancer_predictor,skin_cancer_cnn_predictor]
Worker_with_tools = WorkerPrompt | Worker.bind_tools(
    tools,
    tool_choice="any",
    parallel_tool_calls=True
)