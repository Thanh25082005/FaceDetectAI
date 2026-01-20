"""
API Routes for Face Recognition System

FastAPI endpoints for:
- Face detection
- Face recognition
- Anti-spoofing
- Face database management
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np

from .schemas import (
    DetectFaceResponse, FaceDetection,
    RecognizeFaceResponse, RecognitionMatch,
    AddFaceResponse, GetFaceResponse, UpdateFaceResponse, DeleteFaceResponse,
    HealthResponse, MobileCheckinResponse,
    FASCheckinResponse, FASCheckinStep
)

from models.face_detector import get_face_detector
from models.face_recognizer import get_face_recognizer
from models.anti_spoofing import get_fas_predictor

from models.database import get_face_database
from utils.image_utils import load_image_from_bytes
from utils.geo_utils import calculate_distance
import config
from datetime import datetime
from models.checkin_logger import get_checkin_logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    db = get_face_database()
    
    # Check model availability
    try:
        detector = get_face_detector()
        detector_available = detector.detector is not None
    except Exception:
        detector_available = False
    
    try:
        recognizer = get_face_recognizer()
        recognizer_available = recognizer.app is not None
    except Exception:
        recognizer_available = False
    
    try:
        anti_spoof = get_anti_spoofing()
        anti_spoof_available = True  # Always available now (no mediapipe dependency)
    except Exception:
        anti_spoof_available = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "face_detector": detector_available,
            "face_recognizer": recognizer_available,
            "anti_spoofing": anti_spoof_available
        },
        database_users=db.get_user_count()
    )


@router.post("/detect_face", response_model=DetectFaceResponse)
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image
    
    Returns bounding boxes and facial landmarks for all detected faces.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Detect faces
    detector = get_face_detector()
    
    try:
        detections = detector.detect_faces(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    # Format response
    faces = []
    for det in detections:
        face = FaceDetection(
            box=det['box'],
            confidence=det['confidence'],
            landmarks=det.get('landmarks')
        )
        faces.append(face)
    
    height, width = image.shape[:2]
    
    return DetectFaceResponse(
        success=True,
        faces_count=len(faces),
        faces=faces,
        image_size={"width": width, "height": height}
    )


@router.post("/recognize_face", response_model=RecognizeFaceResponse)
async def recognize_face(
    file: UploadFile = File(...),
    threshold: float = Query(config.FR_THRESHOLD, ge=0.0, le=1.0, description="Similarity threshold for matching")
):
    """
    Recognize faces in an uploaded image by comparing with database
    
    Returns best matches for each detected face.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Get detector, recognizer and database
    detector = get_face_detector()
    recognizer = get_face_recognizer()
    db = get_face_database()
    
    try:
        # Use MTCNN for detection + alignment (optimized pipeline)
        aligned_faces = detector.extract_aligned_faces(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    if not aligned_faces:
        return RecognizeFaceResponse(
            success=True,
            faces_detected=0,
            matches=[],
            message="No faces detected in image"
        )
    
    # Extract embeddings using direct ArcFace model (no redundant detection)
    face_data = []
    for aligned_face, detection in aligned_faces:
        embedding = recognizer.get_embedding_direct(aligned_face)
        if embedding is not None:
            face_data.append({
                'embedding': embedding.tolist(),
                'box': detection['box'],
                'confidence': detection['confidence']
            })
    
    if not face_data:
        return RecognizeFaceResponse(
            success=True,
            faces_detected=len(aligned_faces),
            matches=[],
            message="Failed to extract embeddings from detected faces"
        )
    
    # Get all embeddings from database
    db_embeddings = db.get_all_embeddings()
    
    if not db_embeddings:
        return RecognizeFaceResponse(
            success=True,
            faces_detected=len(face_data),
            matches=[],
            message="No faces in database to compare with"
        )
    
    # Match each detected face
    matches = []
    for face in face_data:
        query_embedding = np.array(face['embedding'])
        result = recognizer.recognize(query_embedding, db_embeddings, threshold=threshold)
        
        if result:
            # Get name from database
            db_face = db.get_face(result['user_id'])
            name = db_face['name'] if db_face else None
            
            matches.append(RecognitionMatch(
                user_id=result['user_id'],
                name=name,
                similarity=result['similarity'],
                is_match=result['is_match'],
                box=face['box']
            ))

    
    return RecognizeFaceResponse(
        success=True,
        faces_detected=len(face_data),
        matches=matches
    )





@router.post("/add_face", response_model=AddFaceResponse)
async def add_face(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    name: Optional[str] = Form(None)
):
    """
    Add a new face to the database
    
    Extracts face embedding from image and stores it with user_id.
    """
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Get detector and recognizer (optimized pipeline)
    detector = get_face_detector()
    recognizer = get_face_recognizer()
    
    try:
        # Use MTCNN for detection + alignment
        aligned_result = detector.get_largest_aligned_face(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    if aligned_result is None:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    aligned_face, detection = aligned_result
    
    # Extract embedding directly (no redundant detection)
    embedding = recognizer.get_embedding_direct(aligned_face)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to extract face embedding")
    
    embedding = embedding.tolist()
    
    # Store in database
    db = get_face_database()
    result = db.add_face(
        user_id=user_id,
        embedding=embedding,
        name=name
    )
    
    if not result['success']:
        raise HTTPException(status_code=409, detail=result['message'])
    
    return AddFaceResponse(
        success=True,
        message=result['message'],
        user_id=user_id,
        id=result.get('id')
    )


@router.get("/get_face/{user_id}", response_model=GetFaceResponse)
async def get_face(user_id: str):
    """
    Get face data for a user by their user_id
    """
    db = get_face_database()
    face = db.get_face(user_id)
    
    if face is None:
        return GetFaceResponse(
            success=False,
            data=None,
            message=f"User {user_id} not found"
        )
    
    return GetFaceResponse(
        success=True,
        data=face
    )


@router.delete("/delete_face/{user_id}", response_model=DeleteFaceResponse)
async def delete_face(user_id: str):
    """
    Delete a face from the database
    """
    db = get_face_database()
    result = db.delete_face(user_id)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['message'])
    
    return DeleteFaceResponse(
        success=True,
        message=result['message']
    )


@router.post("/mobile_checkin", response_model=MobileCheckinResponse)
async def mobile_checkin(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    expected_user_id: str = Form(None) # Optional for backward compatibility, strict for App
):
    """
    Mobile Check-in Endpoint with Geolocation and Face Auth
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Location Validation
    dist = calculate_distance(latitude, longitude, config.COMPANY_LOCATION[0], config.COMPANY_LOCATION[1])
    if dist > config.MAX_CHECKIN_DISTANCE:
         return MobileCheckinResponse(
             success=False, 
             message=f"Location too far ({dist:.0f}m)", 
             distance_meters=dist, 
             timestamp=timestamp
         )
    
    # 2. Face Authentication
    # Read image
    contents = await file.read()
    
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    
    # Get detector and recognizer models
    detector = get_face_detector()
    recognizer = get_face_recognizer()
    
    try:
        # Use MTCNN for detection + alignment
        aligned_result = detector.get_largest_aligned_face(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    if aligned_result is None:
        return MobileCheckinResponse(
            success=False,
            message="No face detected in image",
            distance_meters=dist,
            timestamp=timestamp
        )
    
    aligned_face, detection = aligned_result
    box = detection['box']
    
    # Extract embedding
    embedding = recognizer.get_embedding_direct(aligned_face)

    if embedding is None:
        return MobileCheckinResponse(
            success=False,
            message="Failed to extract face embedding",
            distance_meters=dist,
            box=box,
            timestamp=timestamp
        )
    
    embedding = embedding.tolist()
    
    # Compare with database faces
    db = get_face_database()
    db_embeddings = db.get_all_embeddings()
    
    if not db_embeddings:
        return MobileCheckinResponse(
            success=False,
            message="No faces in database to compare with",
            distance_meters=dist,
            box=box,
            timestamp=timestamp
        )
    
    query_embedding = np.array(embedding)
    
    # Use a higher threshold for check-in for stricter matching
    recognition_threshold = config.FR_THRESHOLD
    
    result = recognizer.recognize(query_embedding, db_embeddings, threshold=recognition_threshold)
    
    if not result or not result['is_match']:
        return MobileCheckinResponse(
            success=False,
            message="Face not recognized or similarity too low",
            distance_meters=dist,
            box=box,
            timestamp=timestamp
        )
    
    # 3. User ID Validation (if provided)
    if expected_user_id and result['user_id'] != expected_user_id:
        return MobileCheckinResponse(
            success=False,
            message=f"Recognized user ID '{result['user_id']}' does not match expected user ID '{expected_user_id}'",
            distance_meters=dist,
            box=box,
            timestamp=timestamp
        )

    match = recognizer.recognize(query_embedding, db_embeddings, threshold=recognition_threshold)
    
    if match and match['is_match']:
         # CHECK IF MATCHED USER IS THE LOGGED IN USER
         if expected_user_id and match['user_id'] != expected_user_id:
              return MobileCheckinResponse(
                  success=False, 
                  message=f"Face mismatch (Not {expected_user_id})", 
                  distance_meters=dist, 
                  box=box,
                  timestamp=timestamp
              )

         # Success! Log it
         user = db.get_face(match['user_id'])
         name = user['name'] if user else "Unknown"
         
         logger = get_checkin_logger()
         logger.log_checkin(
            user_id=match['user_id'],
            name=name,
            camera_id="mobile_app", # Or a specific mobile device ID if available
            location={'latitude': latitude, 'longitude': longitude},
            checkin_type="mobile"
         )

         return MobileCheckinResponse(
             success=True,
             message="Check-in successful",
             user_id=match['user_id'],
             name=name,
             similarity=match['similarity'],
             distance_meters=dist,
             box=box,
             timestamp=timestamp
         )
    else:
         return MobileCheckinResponse(
             success=False, 
             message="Face not recognized", 
             distance_meters=dist, 
             box=box,
             timestamp=timestamp
         )


@router.post("/checkin_fas", response_model=FASCheckinResponse)
async def checkin_fas(
    file: UploadFile = File(...),
    expected_user_id: str = Form(None)
):
    """
    Enhanced Check-in Endpoint with Step-by-Step FAS + Recognition
    """
    steps = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # helper to add steps
    def add_step(name, status, message, score=None):
        steps.append(FASCheckinStep(
            step_name=name,
            status=status,
            message=message,
            score=score
        ))

    # 1. Read image
    contents = await file.read()
    try:
        image = load_image_from_bytes(contents)
    except Exception as e:
        add_step("loading", "failed", f"Invalid image: {str(e)}")
        return FASCheckinResponse(
            success=False,
            message="Image loading failed",
            steps=steps,
            current_step="loading"
        )

    # 2. Face Detection
    add_step("detecting", "pending", "Looking for face...")
    detector = get_face_detector()
    try:
        aligned_result = detector.get_largest_aligned_face(image)
    except Exception as e:
        add_step("detecting", "failed", f"Detection error: {str(e)}")
        return FASCheckinResponse(
            success=False,
            message="Detection failed",
            steps=steps,
            current_step="detecting"
        )
    
    if aligned_result is None:
        add_step("detecting", "failed", "No face detected")
        return FASCheckinResponse(
            success=False,
            message="No face detected",
            steps=steps,
            current_step="detecting"
        )
    
    aligned_face, detection = aligned_result
    box = detection['box'] # [x1, y1, x2, y2]
    add_step("detecting", "success", "Face found", score=detection['confidence'])
    
    # 3. Anti-Spoofing (FAS)
    add_step("anti_spoofing", "pending", "Checking authenticity...")
    fas_predictor = get_fas_predictor()
    try:
        fas_result = fas_predictor.predict(image)
        fas_score = fas_result['score']
        # Be strict for single-image check-in
        is_real = fas_result['is_real'] and fas_score >= config.FAS_ACCEPT_THRESHOLD
        
        add_step("anti_spoofing", "success" if is_real else "failed", 
                 "Real face verified" if is_real else f"Spoof detected (score: {fas_score:.4f})", 
                 score=fas_score)
        
        if not is_real:
            return FASCheckinResponse(
                success=False,
                message="Spoof detected",
                steps=steps,
                current_step="anti_spoofing",
                fas_score=fas_score,
                is_spoof=True,
                box=box
            )
    except Exception as e:
        add_step("anti_spoofing", "error", f"FAS error: {str(e)}")
        return FASCheckinResponse(
            success=False,
            message="FAS process failed",
            steps=steps,
            current_step="anti_spoofing",
            box=box
        )
    
    # 4. Face Recognition
    add_step("recognizing", "pending", "Recognizing person...")
    recognizer = get_face_recognizer()
    db = get_face_database()
    
    try:
        embedding = recognizer.get_embedding_direct(aligned_face)
        if embedding is None:
            add_step("recognizing", "failed", "Failed to extract features")
            return FASCheckinResponse(
                success=False,
                message="Feature extraction failed",
                steps=steps,
                current_step="recognizing",
                fas_score=fas_score,
                box=box
            )
        
        db_embeddings = db.get_all_embeddings()
        if not db_embeddings:
            add_step("recognizing", "failed", "Database empty")
            return FASCheckinResponse(
                success=False,
                message="Database empty",
                steps=steps,
                current_step="recognizing",
                fas_score=fas_score,
                box=box
            )
        
        match = recognizer.recognize(embedding, db_embeddings, threshold=config.FR_THRESHOLD)
        
        if match and match['is_match']:
            user_id = match['user_id']
            similarity = match['similarity']
            
            # Check expected user id if provided
            if expected_user_id and user_id != expected_user_id:
                add_step("recognizing", "failed", f"User mismatch (Not {expected_user_id})")
                return FASCheckinResponse(
                    success=False,
                    message="User mismatch",
                    steps=steps,
                    current_step="recognizing",
                    fas_score=fas_score,
                    similarity=similarity,
                    box=box
                )
            
            # Get user info
            user_info = db.get_face(user_id)
            name = user_info['name'] if user_info else "Unknown"
            
            # Combined confidence (simple mean for now)
            confidence = (fas_score + similarity) / 2
            
            # Success! Log it
            logger = get_checkin_logger()
            logger.log_checkin(
                user_id=user_id,
                camera_id="api_fas",
                confidence=confidence,
                fas_score=fas_score,
                similarity=similarity,
                evidence_frame=image # Save the full image as evidence
            )
            
            add_step("recognizing", "success", f"Recognized as {name}", score=similarity)
            
            return FASCheckinResponse(
                success=True,
                message="Check-in successful",
                steps=steps,
                current_step="complete",
                user_id=user_id,
                name=name,
                fas_score=fas_score,
                similarity=similarity,
                confidence=confidence,
                is_recognized=True,
                box=box
            )
        else:
            add_step("recognizing", "failed", "Person unknown")
            return FASCheckinResponse(
                success=False,
                message="Person not recognized",
                steps=steps,
                current_step="recognizing",
                fas_score=fas_score,
                box=box
            )
            
    except Exception as e:
        add_step("recognizing", "error", f"Recognition error: {str(e)}")
        return FASCheckinResponse(
            success=False,
            message="Recognition failed",
            steps=steps,
            current_step="recognizing",
            fas_score=fas_score,
            box=box
        )




@router.post("/update_face", response_model=UpdateFaceResponse)
async def update_face(
    file: UploadFile = File(None),
    user_id: str = Form(...),
    name: Optional[str] = Form(None)
):
    """
    Update face data for an existing user
    
    Can update embedding (by providing new image) and/or name.
    """
    db = get_face_database()
    
    # Check if user exists
    existing = db.get_face(user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    embedding = None
    
    # If new image provided, extract embedding
    if file is not None and file.filename:
        contents = await file.read()
        
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        try:
            image = load_image_from_bytes(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Optimized pipeline: MTCNN detect + align, then direct embedding
        detector = get_face_detector()
        recognizer = get_face_recognizer()
        
        try:
            aligned_result = detector.get_largest_aligned_face(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        
        if aligned_result is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        aligned_face, _ = aligned_result
        
        if aligned_face is None or aligned_face.size == 0:
            raise HTTPException(status_code=400, detail="Failed to align face")
        
        embedding = recognizer.get_embedding_direct(aligned_face)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract face embedding")
        
        embedding = embedding.tolist()
    
    # Check if anything to update
    if embedding is None and name is None:
        return UpdateFaceResponse(
            success=True,
            message=f"No changes made for user {user_id}"
        )
    
    # Update in database
    result = db.update_face(
        user_id=user_id,
        embedding=embedding,
        name=name
    )
    
    return UpdateFaceResponse(
        success=result['success'],
        message=result['message']
    )


# =============================================================================
# Real-time Streaming Endpoints
# =============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from models.tracker import get_face_tracker
from models.session_manager import get_session_manager
from models.checkin_logger import get_checkin_logger
from streaming.stream_processor import get_stream_processor
import asyncio
import base64
import cv2


@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for real-time video streaming.
    
    Accepts base64-encoded frames and processes them through the pipeline.
    Returns check-in events and track updates.
    """
    await websocket.accept()
    
    processor = get_stream_processor()
    await processor.start()
    
    # Start processing loop in background
    process_task = asyncio.create_task(processor.process_loop())
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                # Decode base64 frame
                frame_b64 = data.get('frame')
                if frame_b64:
                    frame_bytes = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        await processor.add_frame(frame, camera_id)
            
            elif data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
            
            # Send recent events
            events = processor.get_recent_events(5)
            stats = processor.get_stats()
            
            await websocket.send_json({
                'type': 'update',
                'events': events,
                'stats': stats
            })
            
    except WebSocketDisconnect:
        pass
    finally:
        await processor.stop()
        process_task.cancel()


@router.get("/tracks")
async def get_active_tracks():
    """Get all active face tracks"""
    tracker = get_face_tracker()
    tracks = tracker.get_active_tracks()
    
    return {
        'success': True,
        'count': len(tracks),
        'tracks': [t.to_dict() for t in tracks]
    }


@router.get("/sessions")
async def get_active_sessions():
    """Get all active check-in sessions"""
    session_manager = get_session_manager()
    sessions = session_manager.get_all_sessions()
    
    return {
        'success': True,
        'count': len(sessions),
        'sessions': [s.to_dict() for s in sessions]
    }


@router.get("/checkins")
async def get_checkin_history(
    minutes: int = Query(30, description="How far back to look"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    camera_id: Optional[str] = Query(None, description="Filter by camera")
):
    """Get recent check-in history"""
    logger = get_checkin_logger()
    records = logger.get_recent_checkins(minutes, user_id, camera_id)
    
    return {
        'success': True,
        'count': len(records),
        'checkins': [r.to_dict() for r in records]
    }


@router.get("/checkins/today")
async def get_checkins_today(user_id: Optional[str] = None):
    """Get check-in count for today"""
    logger = get_checkin_logger()
    count = logger.get_checkin_count_today(user_id)
    
    return {
        'success': True,
        'count': count,
        'user_id': user_id
    }


@router.get("/stream/stats")
async def get_stream_stats():
    """Get stream processor statistics"""
    processor = get_stream_processor()
    return {
        'success': True,
        'stats': processor.get_stats()
    }

