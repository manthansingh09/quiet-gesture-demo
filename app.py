import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# RTC Configuration for Web Deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def get_intent(hand_landmarks):
    # Rule-based logic for the demo (Stable & Fast)
    lm = hand_landmarks.landmark
    fingers = []
    
    # Thumb
    if lm[4].x < lm[3].x: fingers.append(1)
    else: fingers.append(0)
    
    # Fingers (Index to Pinky)
    for tip in [8, 12, 16, 20]:
        if lm[tip].y < lm[tip-2].y: fingers.append(1)
        else: fingers.append(0)
    
    total = sum(fingers)
    
    if total == 5: return "NEED HELP", (0, 255, 0)
    if total == 0: return "PAIN", (0, 0, 255)
    if total == 2 and fingers[1] and fingers[2]: return "WATER", (255, 255, 0)
    if total == 1 and fingers[1]: return "ATTENTION", (255, 165, 0)
    if lm[8].y < 0.2: return "EMERGENCY", (0, 0, 255)
    
    return "Scanning...", (255, 255, 255)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            intent, color = get_intent(hand_lms)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Draw overlay
            cv2.putText(img, f"INTENT: {intent}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            if intent == "EMERGENCY":
                cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 10)

    return frame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.set_page_config(page_title="Quiet Gesture Demo", layout="wide")
st.title("🏥 Quiet Gesture AI")
st.write("Assistive Communication System for Non-Verbal Patients")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="quiet-gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Gesture Key")
    st.write("🖐 **Open Palm**: NEED HELP")
    st.write("✊ **Fist**: PAIN")
    st.write("✌️ **Two Fingers**: WATER")
    st.write("☝️ **Index Up**: ATTENTION")
    st.write("🙋 **Hand High**: EMERGENCY")
    
    st.divider()
    st.info("The system uses real-time computer vision to map landmarks to patient needs.")
