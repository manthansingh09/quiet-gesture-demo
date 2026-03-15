import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('quiet_gesture_model.h5')

# Gesture Labels
LABELS = ['YES', 'NO', 'NEED HELP', 'PAIN', 'WATER', 'CALL NURSE', 'EMERGENCY']

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

class GestureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Landmark extraction
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        label_text = "Scanning..."
        color = (255, 255, 255)

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Prediction
            prediction = model.predict(np.array([landmarks]))
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.7:
                label_text = f"{LABELS[class_id]} ({int(confidence*100)}%)"
                
                # Visual Alert for Emergency
                if LABELS[class_id] == 'EMERGENCY':
                    color = (0, 0, 255)
                    cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
            
            mp_draw.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(img, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# Streamlit UI
st.set_page_config(page_title="Quiet Gesture AI", layout="centered")
st.title("🏥 Quiet Gesture: Patient AI")
st.markdown("Interpret gestures for non-verbal medical communication.")

ctx = webrtc_streamer(key="gesture-detection", video_transformer_factory=GestureTransformer)

if ctx.video_transformer:
    st.write("### Caregiver Status Dashboard")
    # In a real app, we would use session_state to pass data from the transformer to the UI
    st.info("System Online. Watching for patient movement...")