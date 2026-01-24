from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import route handlers from existing routes
from api.routes import (
    add_face, 
    mobile_checkin, 
    get_face, 
    delete_face, 
    get_config, 
    update_config,
    health_check
)
from config import API_HOST, API_PORT

# Create FastAPI app
app = FastAPI(
    title="Face Recognition Mobile API",
    description="Streamlined API for Mobile Check-in",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Mobile Router
mobile_router = APIRouter()

# Register only the requested endpoints
mobile_router.post("/add_face")(add_face)
mobile_router.post("/mobile_checkin")(mobile_checkin)
mobile_router.get("/get_face/{user_id}")(get_face)
mobile_router.delete("/delete_face/{user_id}")(delete_face)
mobile_router.get("/config")(get_config)
mobile_router.post("/config")(update_config)
mobile_router.get("/health")(health_check)

# Include Router
app.include_router(mobile_router, prefix="/api/v1", tags=["Mobile API"])

@app.get("/")
async def root():
    return {
        "message": "Face Recognition Mobile API",
        "endpoints": [
            "/api/v1/mobile_checkin",
            "/api/v1/add_face",
            "/api/v1/get_face/{user_id}",
            "/api/v1/delete_face/{user_id}",
            "/api/v1/config",
            "/api/v1/health"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_mobile:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
