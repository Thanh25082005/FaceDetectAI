from fastapi import APIRouter, HTTPException, Depends
from models.database import get_face_database
from api.schemas import LoginRequest, LoginResponse, RegisterRequest
import hashlib

router = APIRouter()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@router.post("/auth/register")
async def register(req: RegisterRequest):
    """
    Register a new user account.
    Face ID will be generated here to link with future face enrollment.
    """
    db = get_face_database()
    
    # Check if username exists
    existing = db.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Generate unique face_user_id (to be used when adding face)
    # We use username as base for simplicity, or random UUID
    face_user_id = f"user_{req.username}"
    
    password_hash = hash_password(req.password)
    
    result = db.create_user(
        username=req.username,
        password_hash=password_hash,
        full_name=req.full_name,
        dob=req.dob,
        face_user_id=face_user_id
    )
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result['message'])
        
    return {
        "success": True, 
        "message": "Registration successful", 
        "face_user_id": face_user_id
    }

@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """
    Login with username/password
    """
    db = get_face_database()
    user = db.get_user_by_username(req.username)
    
    if not user:
        return LoginResponse(success=False, message="Invalid username or password")
        
    # Verify password
    if user['password_hash'] != hash_password(req.password):
        return LoginResponse(success=False, message="Invalid username or password")
        
    return LoginResponse(
        success=True,
        message="Login successful",
        face_user_id=user['face_user_id'],
        full_name=user['full_name']
    )
