import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_DETAILS = os.getenv("MONGO_DETAILS")

if not MONGO_DETAILS:
    raise ValueError("MONGO_DETAILS environment variable not set!")

client = MongoClient(MONGO_DETAILS)
database = client.financial_recovery_db # You can name your database

# Get the collection for users
user_collection = database.get_collection("users_collection")
