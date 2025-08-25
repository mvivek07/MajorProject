from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    """Schema for user creation request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str
    company_name: str
    job_title: str
    industry: str
    annual_revenue: str
    primary_financial_goal: str

class UserInDB(UserCreate):
    """Schema for user object stored in the database."""
    hashed_password: str

class UserPublic(BaseModel):
    """Schema for public user information (response model)."""
    email: EmailStr
    full_name: str
    company_name: str
    job_title: str
    
    class Config:
        from_attributes = True # Pydantic V2 compatibility

class Token(BaseModel):
    """Schema for the JWT token response."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for data inside the JWT token."""
    username: str | None = None
