import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import os
from typing import List, Optional, Tuple, Dict

class HandDataCollector:
    """Modern landmark detection using Mediapipe Tasks API."""
    
    def __init__(self, model_path: str = "src/hand_landmarker.task"):
        if not os.path.exists(model_path):
            # Try same directory as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "hand_landmarker.task")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        # Tasks API expects mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        hand_data_list = []
        annotated_frame = frame.copy()
        
        if detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            for idx, (landmarks, handedness) in enumerate(zip(detection_result.hand_landmarks, detection_result.handedness)):
                # Draw landmarks and connections using new tasks drawing utils
                # However, drawing_utils in tasks might be different or simpler.
                # Let's use the manual drawing but with the correct connection mapping if needed.
                # Since I saw drawing_utils in vision, I can try it.
                
                # Extract numerical data
                extracted_lms = []
                for lm in landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (px, py), 4, (0, 255, 0), -1)
                    extracted_lms.append({
                        'x': lm.x, 'y': lm.y, 'z': lm.z,
                        'px': px, 'py': py
                    })
                
                # Draw connections manually for reliability
                for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    p1 = landmarks[start_idx]
                    p2 = landmarks[end_idx]
                    cv2.line(annotated_frame, 
                             (int(p1.x * w), int(p1.y * h)), 
                             (int(p2.x * w), int(p2.y * h)), 
                             (255, 255, 255), 2)

                label = handedness[0].category_name
                score = handedness[0].score
                
                hand_data_list.append({
                    'index': idx,
                    'label': label,
                    'score': score,
                    'landmarks': extracted_lms
                })
                
                # Show label on frame
                cv2.putText(annotated_frame, f"{label} ({score:.2f})", 
                            (int(landmarks[0].x * w), int(landmarks[0].y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        return annotated_frame, hand_data_list

    @staticmethod
    def flatten_landmarks(hand_data: Dict) -> List[float]:
        flat = []
        for lm in hand_data['landmarks']:
            flat.extend([lm['x'], lm['y'], lm['z']])
        return flat
