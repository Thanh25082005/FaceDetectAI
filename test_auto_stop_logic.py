#!/usr/bin/env python3
"""
Test script to verify auto-stop logic
"""
import requests
import cv2
import sys

API_URL = "http://localhost:8000/api/v1"

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return False

def test_recognition(image_path=None):
    """Test face recognition with webcam or image"""
    print("\n🔍 Testing face recognition...")
    
    if image_path:
        # Use provided image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            return False
    else:
        # Capture from webcam
        print("📹 Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        print("⏳ Capturing frame in 3 seconds...")
        for i in range(90):  # ~3 seconds at 30fps
            ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            print("❌ Failed to capture frame")
            return False
    
    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    # Test recognition
    try:
        response = requests.post(
            f"{API_URL}/recognize_face?threshold=0.5",
            files={'file': ('test.jpg', image_bytes, 'image/jpeg')},
            timeout=10
        )
        result = response.json()
        
        print(f"\n📊 Recognition Result:")
        print(f"Success: {result.get('success')}")
        
        if result.get('success'):
            matches = result.get('matches', [])
            if matches:
                for i, match in enumerate(matches):
                    print(f"\nMatch {i+1}:")
                    print(f"  User ID: {match.get('user_id')}")
                    print(f"  Name: {match.get('name')}")
                    print(f"  Similarity: {match.get('similarity', 0):.2%}")
                    print(f"  Is Match: {match.get('is_match')}")
                    
                # Check if ANY match succeeded
                has_match = any(m.get('is_match') for m in matches)
                if has_match:
                    print("\n✅ RECOGNITION SUCCESS - Should AUTO-STOP!")
                    return True
                else:
                    print("\n⚠️  Face detected but NOT recognized - Should CONTINUE scanning")
                    return False
            else:
                print("⚠️  No matches found")
                return False
        else:
            print(f"❌ API Error: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_checkin(image_path=None):
    """Test full pipeline check-in"""
    print("\n⚡ Testing check-in pipeline...")
    
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            return False
    else:
        print("📹 Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        print("⏳ Capturing frame in 3 seconds...")
        for i in range(90):
            ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            print("❌ Failed to capture frame")
            return False
    
    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    # Test pipeline
    try:
        response = requests.post(
            f"{API_URL}/process_frame",
            files={'file': ('test.jpg', image_bytes, 'image/jpeg')},
            data={'camera_id': 'test'},
            timeout=10
        )
        result = response.json()
        
        print(f"\n📊 Pipeline Result:")
        print(f"Success: {result.get('success')}")
        
        if result.get('success'):
            sessions = result.get('sessions', [])
            if sessions:
                session = sessions[0]
                decision = session.get('decision')
                user_id = session.get('matched_user_id')
                confidence = session.get('decision_confidence', 0)
                
                print(f"\nSession Info:")
                print(f"  Decision: {decision}")
                print(f"  User ID: {user_id}")
                print(f"  Confidence: {confidence:.2%}")
                
                if decision == 'accepted':
                    print("\n✅ CHECK-IN ACCEPTED - Should AUTO-STOP!")
                    return True
                else:
                    print(f"\n⚠️  Decision: {decision} - Should CONTINUE scanning")
                    return False
            else:
                print("⚠️  No sessions found")
                return False
        else:
            print(f"❌ API Error: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def check_database():
    """Check how many faces in database"""
    print("\n📊 Checking database...")
    try:
        response = requests.get(f"{API_URL}/faces", timeout=5)
        if response.status_code == 200:
            faces = response.json()
            print(f"✅ Database has {len(faces)} faces:")
            for face in faces:
                print(f"  - {face.get('name')} ({face.get('user_id')})")
            return len(faces) > 0
        else:
            print(f"❌ Cannot get faces: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == '__main__':
    print("="*50)
    print("🧪 AUTO-STOP LOGIC TEST")
    print("="*50)
    
    # Test 1: API Health
    if not test_api_health():
        print("\n❌ API is not running. Start it first:")
        print("   uvicorn main:app --reload --port 8000")
        sys.exit(1)
    
    # Test 2: Database
    if not check_database():
        print("\n⚠️  No faces in database!")
        print("   Use Enrollment mode to add faces first")
    
    # Test 3: Recognition
    print("\n" + "="*50)
    if test_recognition():
        print("\n✅ Recognition test PASSED - Auto-stop should work!")
    else:
        print("\n⚠️  Recognition test FAILED - Will continue scanning")
    
    # Test 4: Check-in Pipeline
    print("\n" + "="*50)
    if test_checkin():
        print("\n✅ Check-in test PASSED - Auto-stop should work!")
    else:
        print("\n⚠️  Check-in test FAILED - Will continue scanning")
    
    print("\n" + "="*50)
    print("🎯 CONCLUSION:")
    print("="*50)
    print("""
If tests PASSED:
  → Auto-stop WILL work in demo_pipeline and demo_realtime
  → Open http://localhost:8505 to test

If tests FAILED:
  → Either no face in database OR
  → Face quality too low OR  
  → Anti-spoofing rejecting (using photo instead of real face)
  
To fix:
  1. Add face using Enrollment mode first
  2. Use real face (not photo)
  3. Good lighting
  4. Look straight at camera
""")
