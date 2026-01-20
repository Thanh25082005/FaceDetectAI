#!/usr/bin/env python3
"""
Quick script to add a face to database
Usage: python quick_add_face.py <user_id> <name>
"""
import sys
import cv2
import requests
import time

API_URL = "http://localhost:8000/api/v1"

def add_face_to_db(user_id, name):
    """Capture face from webcam and add to database"""
    print(f"\n📹 Opening webcam to capture face for {name} ({user_id})...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("⏳ Get ready! Capturing in 3 seconds...")
    print("   - Look straight at camera")
    print("   - Good lighting")
    print("   - Remove glasses if possible")
    
    # Wait and capture
    for i in range(90):  # ~3 seconds
        ret, frame = cap.read()
        if i % 30 == 0:
            print(f"   {3 - i//30}...")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to capture frame")
        return False
    
    print("✅ Frame captured! Sending to API...")
    
    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    # Send to API
    try:
        response = requests.post(
            f"{API_URL}/add_face",
            files={'file': ('face.jpg', image_bytes, 'image/jpeg')},
            data={'user_id': user_id, 'name': name},
            timeout=10
        )
        
        result = response.json()
        
        if result.get('success'):
            print(f"\n🎉 SUCCESS! Added {name} ({user_id}) to database")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            return True
        else:
            print(f"\n❌ FAILED: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python quick_add_face.py <user_id> <name>")
        print("Example: python quick_add_face.py user001 'Nguyen Van A'")
        sys.exit(1)
    
    user_id = sys.argv[1]
    name = sys.argv[2]
    
    print("="*50)
    print("🚀 QUICK ADD FACE TO DATABASE")
    print("="*50)
    
    # Check API
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        print("✅ API is running\n")
    except:
        print("❌ API not running! Start it first:")
        print("   uvicorn main:app --reload --port 8000")
        sys.exit(1)
    
    if add_face_to_db(user_id, name):
        print("\n✅ Now you can test auto-stop at http://localhost:8505")
        print("   Use 'Chấm công (Check-in)' mode")
    else:
        print("\n❌ Failed to add face. Try again with:")
        print("   - Better lighting")
        print("   - Look straight at camera")
        print("   - No glasses")
