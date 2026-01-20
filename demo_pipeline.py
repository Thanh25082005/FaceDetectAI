#!/usr/bin/env python3
"""
Streamlit Smart Kiosk v2

Logic: Detect Face → Stop Stream → Process
"""

import streamlit as st
import requests
import cv2
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Quick Face Kiosk", page_icon="⚡", layout="wide")
API_URL = "http://localhost:8000/api/v1"

def check_server():
    try:
        requests.get(f"{API_URL}/health", timeout=1)
        return True
    except:
        return False

def detect_only_local(frame):
    """Simple OpenCV detection just to check if face exists (Faster than API roundtrip)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0, faces

def process_checkin(frame):
    """Send frame to server for recognition (simple endpoint)"""
    _, buf = cv2.imencode('.jpg', frame)
    try:
        # Use recognize_face - simple recognition endpoint
        resp = requests.post(f"{API_URL}/recognize_face", 
                           files={'file': ('f.jpg', buf.tobytes(), 'image/jpeg')},
                           params={'threshold': 0.5}, timeout=2)
        return resp.json()
    except:
        return None

def process_enroll(frame, uid, name):
    """Send frame to server for enrollment"""
    _, buf = cv2.imencode('.jpg', frame)
    try:
        resp = requests.post(f"{API_URL}/add_face",
                           files={'file': ('f.jpg', buf.tobytes(), 'image/jpeg')},
                           data={'user_id': uid, 'name': name}, timeout=5)
        return resp.json()
    except:
        return {'success': False, 'message': 'API Error'}

def main():
    st.title("⚡ Quick Detect & Process")
    
    mode = st.radio("Chế độ:", ["Chấm công (Check-in)", "Đăng ký (Enrollment)"], horizontal=True)
    
    # Init session state
    if 'stop_stream' not in st.session_state:
        st.session_state['stop_stream'] = False
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        start = st.button("▶️ BẮT ĐẦU", type="primary", use_container_width=True)
        video_place = st.empty()
        
    with col2:
        result_place = st.empty()
        
        # Enrollment inputs
        if "Đăng ký" in mode:
            st.divider()
            uid = st.text_input("User ID")
            name = st.text_input("Name")
            if not uid or not name:
                st.info("Nhập thông tin để đăng ký")

    if start:
        if "Đăng ký" in mode and (not uid or not name):
            st.warning("Thiếu thông tin User ID/Name")
            return
            
        cap = cv2.VideoCapture(0)
        st.session_state['stop_stream'] = False
        
        frame_count = 0
        process_frame_interval = 2  # Process every 2nd frame (50% reduction)
        status_text = st.empty()
        
        while not st.session_state['stop_stream']:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # Show Feed (always display for smooth video)
            video_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            status_text.text(f"Đang tìm khuôn mặt... Frame {frame_count}")
            
            # SKIP PROCESSING for alternate frames to reduce load
            if frame_count % process_frame_interval != 0:
                time.sleep(0.01)
                continue
            
            # --- LOGIC MỚI: DETECT LOCAL → NHẬN DIỆN → CHỈ DỪNG KHI THÀNH CÔNG ---
            has_face, faces = detect_only_local(frame)
            
            if has_face:
                # Tìm thấy mặt -> Thử nhận diện TRƯỚC
                status_text.info("🔍 Phát hiện khuôn mặt! Đang nhận diện...")
                
                # Draw box on current frame
                temp_frame = frame.copy()
                for (x,y,w,h) in faces:
                    cv2.rectangle(temp_frame, (x,y), (x+w,y+h), (0,255,0), 2)
                video_place.image(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB), caption="Đang xử lý...", channels="RGB", use_container_width=True)
                
                # Xử lý frame
                recognition_success = False
                
                if "Chấm công" in mode:
                    res = process_checkin(frame)
                    if res and res.get('success'):
                        matches = res.get('matches', [])
                        
                        # Check if any match found
                        if matches:
                            # Get first match (highest similarity)
                            match = matches[0]
                            
                            if match.get('is_match'):
                                # THÀNH CÔNG -> DỪNG
                                recognition_success = True
                                st.session_state['stop_stream'] = True
                                
                                user = match.get('name') or match.get('user_id')
                                similarity = match.get('similarity', 0)
                                
                                result_place.success(f"✅ CHẤM CÔNG THÀNH CÔNG!\nXin chào: **{user}**\nSimilarity: {similarity:.2%}")
                                video_place.image(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB), caption="✅ Đã nhận diện thành công!", channels="RGB", use_container_width=True)
                                st.balloons()
                            else:
                                # KHÔNG MATCH -> TIẾP TỤC QUÉT
                                similarity = match.get('similarity', 0)
                                status_text.warning(f"⚠️ Similarity quá thấp ({similarity:.2%} < 50%). Tiếp tục quét...")
                        else:
                            # KHÔNG TÌM THẤY MATCH -> TIẾP TỤC
                            status_text.warning("⚠️ Không nhận diện được. Tiếp tục quét...")
                    else:
                        status_text.warning("⚠️ Không tìm thấy khuôn mặt hợp lệ. Tiếp tục quét...")
                        
                else: # Enrollment
                    res = process_enroll(frame, uid, name)
                    if res.get('success'):
                        # ĐĂNG KÝ THÀNH CÔNG -> DỪNG
                        recognition_success = True
                        st.session_state['stop_stream'] = True
                        result_place.success(f"✅ ĐĂNG KÝ THÀNH CÔNG!\n{name} ({uid})")
                        video_place.image(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB), caption="✅ Đã đăng ký thành công!", channels="RGB", use_container_width=True)
                        st.balloons()
                    else:
                        # ĐĂNG KÝ THẤT BẠI -> TIẾP TỤC QUÉT
                        status_text.warning(f"⚠️ Đăng ký thất bại: {res.get('message')}. Tiếp tục quét...")
                
                # Chỉ dừng camera nếu thành công
                if recognition_success:
                    cap.release()
                    
            time.sleep(0.01) # Low latency loop
            
        if cap.isOpened():
            cap.release()

if __name__ == '__main__':
    main()
