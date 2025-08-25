from .database import user_collection
from .schemas import UserCreate
from .security import get_password_hash
from bson import ObjectId

async def get_user_by_email(email: str):
    """Fetches a single user from the database by their email."""
    return user_collection.find_one({"email": email})

async def create_user(user: UserCreate):
    """Creates a new user in the database."""
    hashed_password = get_password_hash(user.password)
    user_data = user.model_dump()
    user_data.pop("password") # Remove plain password
    
    user_db_object = {
        **user_data,
        "hashed_password": hashed_password
    }
    
    result = user_collection.insert_one(user_db_object)
    new_user = user_collection.find_one({"_id": result.inserted_id})
    return new_user