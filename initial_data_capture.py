import cv2
import face_recognition
import os
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
    At first, a person's image was taken using a reference object.     
    
    args:
    camera_id : int
    """
    path = "Attendance_data/"
    if camera_id == None:
        camera_id = 0  # Use default camera on Windows
    
    # Check existing names in the Attendance_data folder
    existing_names = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            name_without_ext = os.path.splitext(filename)[0]
            existing_names.append(name_without_ext.lower())  # Store names in lowercase
    
    while True:
        Name = input("Please Enter your name: ")
        if Name.lower() in existing_names:
            print(f"Error: {Name} already exists in the database!")
            retry = input("Do you want to try a different name? (yes/no): ")
            if retry.lower() != 'yes':
                print("Registration cancelled.")
                return
        else:
            break

    camera = cv2.VideoCapture(camera_id)
    frame_count = 0
    PROCESS_EVERY_N = 5  # hanya proses setiap 5 frame

    while True:
        ret, image = camera.read()
        frame_count += 1
        if frame_count % PROCESS_EVERY_N != 0:
            cv2.imshow('Face Capture', image)
            if cv2.waitKey(1) == 27:
                break
            continue
    
    # Constants for detection
    EYE_AR_THRESH = 0.25        # Adjusted threshold for blink detection
    EYE_AR_CONSEC_FRAMES = 2    # Number of consecutive frames to ensure it's a blink
    
    # Initialize variables
    blink_counter = 0           # Count the number of consecutive frames with closed eyes
    blink_detected = False
    last_ear = 0.30            # Initialize last EAR for comparison
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
            
        display_image = image.copy()
        
        # Convert to RGB for face_recognition (reduced size for speed)
        small_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        
        if len(face_locations) > 0:
            face_landmarks = face_recognition.face_landmarks(rgb_small)
            
            if face_landmarks:
                landmarks = face_landmarks[0]
                
                # Calculate eye aspect ratios
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Check for blink
                if avg_ear < EYE_AR_THRESH:
                    blink_counter += 1
                    if blink_counter >= EYE_AR_CONSEC_FRAMES and not blink_detected:
                        blink_detected = True
                        cv2.putText(display_image, "Blink Detected!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    blink_counter = 0
                
                # Draw measurements and status
                cv2.putText(display_image, f"EAR: {avg_ear:.2f}", (300, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Counter: {blink_counter}", (300, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # If blink detected, save image
                if blink_detected:
                    # Wait a moment to capture open eyes
                    if avg_ear > 0.28:  # Make sure eyes are open again
                        cv2.imwrite(f'{path}{Name}.png', image)
                        print(f"Image saved successfully! (Blink detected)")
                        break
                
                # Draw face rectangle
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_image, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display_image, "Blink to capture", (10, 450),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow('Face Capture', display_image)
        
        # ESC to exit
        if cv2.waitKey(1) == 27:
            print("Capture cancelled")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    del(camera)
    
Intial_data_capture()
