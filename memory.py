from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
import os 
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client   = MongoClient(MONGO_URI)
DB       = client["viora_ai"]

checkpointer = MongoDBSaver(DB)