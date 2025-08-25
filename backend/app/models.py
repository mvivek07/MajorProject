from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from bson import ObjectId

class PyObjectId(ObjectId):
    """ Custom Pydantic field for MongoDB's ObjectId. """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class User(BaseModel):
    """
    Represents a user document as stored in the MongoDB collection.
    This model includes all user data, including the hashed password.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr
    hashed_password: str
    full_name: str
    company_name: str
    job_title: str
    industry: str
    annual_revenue: str
    primary_financial_goal: str

    class Config:
        """
        Pydantic configuration to handle MongoDB's '_id' field
        and allow population by field name.
        """
        populate_by_name = True
        arbitrary_types_allowed = True # Needed for ObjectId
        json_encoders = {ObjectId: str}

