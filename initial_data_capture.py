import cv2
import face_recognition
import os

import cv2
import face_recognition
import numpy as np

def calculate_eye_aspect_ratio(eye_landmarks):
    """Calculate the eye aspect ratio to detect blinks"""
    # Compute distances between vertical eye landmarks
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    # Compute distance between horizontal eye landmarks
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    # Calculate eye aspect ratio
    ear = (v1 + v2) / (2.0 * h)
    return ear

def Intial_data_capture(camera_id=None):
    """
    Capture a face image when detecting eye blink.
    
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
    
    blink_detected = False
    
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
            
            # Detect blink
            if avg_ear < EYE_AR_THRESH:
                blink_detected = True
                cv2.putText(display_image, "Blink Detected!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw status
            cv2.putText(display_image, f"Eye AR: {avg_ear:.2f}", (300, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If blink detected, save the image
            if blink_detected:
                cv2.imwrite(f'{path}{Name}.png', image)
                print(f"Image saved successfully!")
                break
        
        # Show instructions
        cv2.putText(display_image, "Blink eyes to capture", (10, 450),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow('Capturing', display_image)
        
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
