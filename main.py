from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router
from api.auth import router as auth_router
from config import API_HOST, API_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan events for startup and shutdown.
    
    Startup:
        - Initialize database connection
        - Create tables if not exist
        - Load face embeddings into memory cache
    
    Shutdown:
        - Close database connection
    """
    print("🚀 Starting Face Recognition API...")
    
    # === STARTUP ===
    try:
        # Initialize SQL Server database
        from database.connection import init_database, test_connection
        
        # Test connection first
        if await test_connection():
            # Create tables
            await init_database()
            
            # Load embeddings into memory for fast recognition
            from services.face_recognition import get_face_service
            face_service = get_face_service()
            await face_service.load_embeddings_to_memory()
            
            print("✅ Face Recognition API ready!")
        else:
            print("⚠️ SQL Server not available. Running without cache.")
            
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("   API will start but some features may not work.")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    print("🔌 Shutting down...")
    try:
        from database.connection import close_database
        await close_database()
    except Exception as e:
        print(f"⚠️ Shutdown warning: {e}")
    
    print("👋 Goodbye!")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Face Recognition API",
    description="""
    Face Recognition Check-in System API
    
    Features:
    - Face detection and recognition
    - Anti-spoofing protection
    - GPS-based check-in verification
    - Real-time video streaming
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Face Recognition"])
app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/cache/status")
async def cache_status():
    """Get face recognition cache status"""
    try:
        from services.face_recognition import get_face_service
        service = get_face_service()
        return service.get_status()
    except Exception as e:
        return {"error": str(e), "is_loaded": False}


@app.post("/cache/refresh")
async def cache_refresh():
    """Manually refresh the face embeddings cache"""
    try:
        from services.face_recognition import get_face_service
        service = get_face_service()
        count = await service.refresh_cache()
        return {"success": True, "faces_loaded": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
