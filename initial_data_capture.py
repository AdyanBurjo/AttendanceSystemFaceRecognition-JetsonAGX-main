import cv2
import face_recognition
import os
import numpy as np

def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Calculate the eye aspect ratio to detect blinks
    """
    # Calculate the euclidean distances
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    # Calculate eye aspect ratio
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate the mouth aspect ratio to detect open mouth
    """
    # Calculate the euclidean distances
    vertical = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
    horizontal = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
    
    # Calculate mouth aspect ratio
    mar = vertical / horizontal
    return mar

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
    
    # Constants for blink and mouth detection
    EYE_BLINK_THRESHOLD = 0.3  # Lower value means more closed
    MOUTH_OPEN_THRESHOLD = 0.5  # Higher value means more open
    
    blink_counter = 0
    mouth_counter = 0
    is_mouth_open = False
    
    print("\nInstructions:")
    print("1. Look at the camera")
    print("2. Blink your eyes and open your mouth")
    print("3. Hold still until capture is complete")
    print("Press ESC to cancel\n")
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to grab frame")
            break
            
        # Convert to RGB for face_recognition
        small_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_landmarks = face_recognition.face_landmarks(rgb_small)

        
        # Create copy for drawing
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
            
            # Draw facial landmarks for visualization
            for feature in landmarks.values():
                points = np.array(feature)
                cv2.polylines(display_image, [points], True, (0, 255, 0), 1)
            
            # Check for blink and open mouth
            if avg_ear < EYE_BLINK_THRESHOLD:
                blink_counter += 1
                cv2.putText(display_image, "BLINK DETECTED!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if mar > MOUTH_OPEN_THRESHOLD:
                is_mouth_open = True
                mouth_counter += 1
                cv2.putText(display_image, "MOUTH OPEN!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display current values
            cv2.putText(display_image, f"EAR: {avg_ear:.2f}", (300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_image, f"MAR: {mar:.2f}", (300, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # If both blink and mouth open are detected
            if blink_counter >= 3 and is_mouth_open:
                cv2.putText(display_image, "CAPTURING!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Save image
                cv2.imwrite(f'{path}{Name}'+'.png', image)
                print(f"Image saved successfully with face detected!")
                break
        
        # Show the image
        cv2.imshow('Capture - Blink and Open Mouth', display_image)
        
        # Check for ESC key
        if cv2.waitKey(1) == 27:  # ESC
            print("Capture cancelled")
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    
Intial_data_capture()
