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
    EYE_AR_THRESH = 0.19  # Threshold for blink detection (lower = more sensitive)
    CHECK_INTERVAL = 0.2   # Check every 0.2 seconds for better performance
    
    blink_detected = False
    last_check_time = 0
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
            
        display_image = image.copy()
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Process every CHECK_INTERVAL seconds
        if current_time - last_check_time >= CHECK_INTERVAL:
            last_check_time = current_time
            
            # Convert to RGB for face_recognition
            small_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_landmarks = face_recognition.face_landmarks(rgb_small)

            
            if len(face_locations) > 0:
                
                if face_landmarks:
                    landmarks = face_landmarks[0]
                    
                    # Calculate eye aspect ratios
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    left_ear = calculate_eye_aspect_ratio(left_eye)
                    right_ear = calculate_eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Update detection flags
                    if avg_ear < EYE_AR_THRESH:
                        blink_detected = True
                        cv2.putText(display_image, "Blink Detected!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw measurements
                    cv2.putText(display_image, f"EAR: {avg_ear:.2f}", (300, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # If both actions detected, save image
                    if blink_detected:
                        cv2.imwrite(f'{path}{Name}.png', image)
                        print(f"Image saved successfully! (Blink and mouth opening detected)")
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
