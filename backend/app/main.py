from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import users

app = FastAPI(
    title="AI Financial Recovery API",
    description="API for the AI-driven debt optimization and financial recovery system.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your React frontend (running on a different port) to communicate with the API.
# IMPORTANT: For production, you should restrict the origins to your actual frontend domain.
origins = [
    "http://localhost:3000", # Default React dev server
    "http://localhost:5173", # Default Vite dev server
    # Add your frontend's production URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
# This makes the user registration and login endpoints available.
app.include_router(users.router)

@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Financial Recovery API!"}
