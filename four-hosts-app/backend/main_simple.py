#!/usr/bin/env python3
"""
Simple backend for testing - no authentication on registration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn
import uuid

app = FastAPI(title="Four Hosts Research API - Simple")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

# Simple in-memory storage
users = {}

@app.get("/")
async def root():
    return {"message": "Four Hosts Research API - Simple", "version": "1.0.0"}

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user"""
    if user_data.email in users:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Store user (in production, hash the password!)
    user_id = str(uuid.uuid4())
    users[user_data.email] = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password": user_data.password  # Don't do this in production!
    }
    
    # Create a simple token
    token = f"token_{user_id}_{uuid.uuid4().hex[:8]}"
    
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=3600
    )

@app.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login with email and password"""
    user = users.get(login_data.email)
    if not user or user["password"] != login_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create a simple token
    token = f"token_{user['id']}_{uuid.uuid4().hex[:8]}"
    
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=3600
    )

@app.get("/auth/user")
async def get_current_user():
    """Get current user (mock)"""
    return {
        "id": "user_123",
        "username": "testuser",
        "email": "test@example.com",
        "role": "free"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)