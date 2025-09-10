import cv2
import face_recognition
import os

import cv2
import face_recognition
import time

import cv2
import face_recognition
import numpy as np
import time

def calculate_eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio for blink detection"""
    # Compute vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Compute horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    # Return aspect ratio
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def Intial_data_capture(camera_id=None):
    """
    Capture a face image when detecting eye blink and mouth opening.
    
    args:
    camera_id : int
    """
    path = "Attendance_data/"
    if camera_id == None:
        camera_id = 0  # Use default camera on Windows
    
    Name = input("Please Enter your name: ")

    camera = cv2.VideoCapture(camera_id)
    
    # Constants for detection
    EYE_AR_THRESH = 0.2  # Threshold for eye blink detection
    MOUTH_AR_THRESH = 0.5  # Threshold for mouth opening detection
    
    blink_detected = False
    mouth_opened = False
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
            
        # Convert to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces and facial landmarks
        face_locations = face_recognition.face_locations(rgb_image)
        face_landmarks = face_recognition.face_landmarks(rgb_image)
        
        # Create a copy for drawing
        display_image = image.copy()
        
        if len(face_locations) > 0 and len(face_landmarks) > 0:
            landmarks = face_landmarks[0]
            
            # Get eye landmarks
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Calculate eye aspect ratios
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate mouth aspect ratio
            mouth = landmarks['top_lip'] + landmarks['bottom_lip']
            mar = calculate_mouth_aspect_ratio(mouth)
            
            # Detect blink
            if avg_ear < EYE_AR_THRESH:
                blink_detected = True
                cv2.putText(display_image, "Blink Detected!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect mouth opening
            if mar > MOUTH_AR_THRESH:
                mouth_opened = True
                cv2.putText(display_image, "Mouth Opened!", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw status
            cv2.putText(display_image, f"Eye AR: {avg_ear:.2f}", (300, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Mouth AR: {mar:.2f}", (300, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If both blink and mouth opening detected, save the image
            if blink_detected and mouth_opened:
                cv2.imwrite(f'{path}{Name}.png', image)
                print(f"Image saved successfully!")
                break
        
        # Show instructions
        cv2.putText(display_image, "Blink eyes and open mouth to capture", (10, 450),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow('Capture - Blink and Open Mouth', display_image)
        
        # ESC to exit
        if cv2.waitKey(1) == 27:
            print("Capture cancelled")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    del(camera)
    camera = cv2.VideoCapture(camera_id)
    
    while True:
        return_value, image = camera.read()
        # Show the image
        cv2.imshow('Capture - Press SPACE when ready', image)
        
        # Wait for spacebar press
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Check if face is detected
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) > 0:
                # Save image if face is detected
                cv2.imwrite(f'{path}{Name}'+'.png', image)
                print(f"Image saved successfully with face detected!")
                break
            else:
                print("No face detected in image. Please try again.")
        
        elif key == 27:  # ESC to exit
            print("Capture cancelled")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    del(camera)
    
Intial_data_capture()
