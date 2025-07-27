#!/usr/bin/env python3
"""
Four Hosts Research API - Development Version with Authentication
This version includes basic authentication endpoints for development/testing
"""

import os
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
from enum import Enum
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core services (simplified for development)
try:
    from services.research_orchestrator import (
        research_orchestrator,
        initialize_research_system,
        execute_research
    )
    from services.cache import initialize_cache
    from services.credibility import get_source_credibility
    from services.llm_client import initialize_llm_client
    from services.answer_generator_continued import answer_orchestrator
    RESEARCH_FEATURES = True
except ImportError as e:
    RESEARCH_FEATURES = False
    logger.warning(f"Research features not available: {e}")

# Data Models
class Paradigm(str, Enum):
    DOLORES = "dolores"
    TEDDY = "teddy"
    BERNARD = "bernard"
    MAEVE = "maeve"

class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

class UserRole(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class ResearchOptions(BaseModel):
    depth: ResearchDepth = ResearchDepth.STANDARD
    paradigm_override: Optional[Paradigm] = None
    include_secondary: bool = True
    max_sources: int = Field(default=50, ge=10, le=200)
    language: str = "en"
    region: str = "us"
    enable_real_search: bool = True

class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    options: ResearchOptions = ResearchOptions()

class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    role: UserRole = UserRole.FREE

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int

class ParadigmClassification(BaseModel):
    primary: Paradigm
    secondary: Optional[Paradigm]
    distribution: Dict[str, float]
    confidence: float
    explanation: Dict[str, str]

class ResearchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    credibility_score: float
    published_date: Optional[str] = None
    source_type: str = "web"

class ResearchResult(BaseModel):
    research_id: str
    query: str
    status: ResearchStatus
    paradigm_analysis: Dict[str, Any]
    answer: Dict[str, Any]
    sources: List[SourceResult]
    metadata: Dict[str, Any]
    cost_info: Optional[Dict[str, float]] = None

# Simple in-memory storage for development
users_db: Dict[str, Dict] = {}
research_store: Dict[str, Dict] = {}
system_initialized = False

# Application Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global system_initialized

    # Startup
    logger.info("🚀 Starting Four Hosts Research API (Development Mode)...")

    try:
        if RESEARCH_FEATURES:
            # Initialize cache system
            cache_success = await initialize_cache()
            if cache_success:
                logger.info("✓ Cache system initialized")

            # Initialize research system
            await initialize_research_system()
            logger.info("✓ Research orchestrator initialized")

            # Initialize LLM client
            await initialize_llm_client()
            logger.info("✓ LLM client initialized")

        system_initialized = True
        logger.info("🚀 Four Hosts Research System ready!")

    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        system_initialized = False

    yield

    # Shutdown
    logger.info("🛑 Shutting down Four Hosts Research API...")

# Create FastAPI App
app = FastAPI(
    title="Four Hosts Research API",
    version="2.0.0-dev",
    description="Development version with authentication support",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID Middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# JWT Configuration
SECRET_KEY = "dev-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password (simplified for development)"""
    return plain_password == hashed_password  # In production, use bcrypt

def hash_password(password: str) -> str:
    """Hash password (simplified for development)"""
    return password  # In production, use bcrypt

# Authentication Endpoints
@app.post("/auth/register", response_model=Token, tags=["authentication"])
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    if any(u["email"] == user_data.email for u in users_db.values()):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user_id = str(uuid.uuid4())
    user = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "password": hash_password(user_data.password),
        "role": user_data.role,
        "created_at": datetime.utcnow().isoformat()
    }

    users_db[user_id] = user

    # Create token
    access_token = create_access_token({
        "user_id": user_id,
        "email": user_data.email,
        "role": user_data.role
    })

    return Token(
        access_token=access_token,
        refresh_token=None,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/auth/login", response_model=Token, tags=["authentication"])
async def login(login_data: UserLogin):
    """Login with email and password"""
    # Find user by email
    user = None
    for u in users_db.values():
        if u["email"] == login_data.email:
            user = u
            break

    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create token
    access_token = create_access_token({
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"]
    })

    return Token(
        access_token=access_token,
        refresh_token=None,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# Root Endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Four Hosts Research API (Development Mode)",
        "version": "2.0.0-dev",
        "system_initialized": system_initialized,
        "features": {
            "research": RESEARCH_FEATURES,
            "authentication": True
        }
    }

# Health Endpoints
@app.get("/health")
async def health_check():
    """System health check"""
    return {
