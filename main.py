from Model import Manager
from Prompt import ManagerPrompt
from langchain_core.messages import HumanMessage,AIMessage
from Graph import Workflow
import os
from anonymizer import process_input
from dotenv import load_dotenv
import uuid

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "viora-ai"


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Session ID: {thread_id}")

    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit", "stop", "end", "close", "goodbye", "bye"]:
            print("Exiting the program.")
            break

        clean_input = process_input(user_input)

        final_answer = Workflow.invoke(
            {
                "whole_messages": [HumanMessage(content=clean_input)],
                "instruction": "",
                "query": "",
                "task": "",     
                "image_type": "", 
                "output": "",
                "messages": []    
            },
            config=config
        )
        print("Final Answer:", final_answer["output"])