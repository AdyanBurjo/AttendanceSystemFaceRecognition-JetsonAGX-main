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
    Capture a face image when detecting eye blinks.
    Optimized for Jetson Nano performance.
    
    args:
    camera_id : int
    """
    path = "Attendance_data/"
    if camera_id == None:
        camera_id = 1  # Use default camera on Windows
    
    Name = input("Please Enter your name: ")

    camera = cv2.VideoCapture(camera_id)
    
    # Constants
    EYE_AR_THRESH = 0.2  # Threshold for blink detection
    EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames for blink
    
    # Variables
    frame_count = 0
    blink_counter = 0
    last_check_time = time.time()
    CHECK_INTERVAL = 0.2  # Check every 0.2 seconds
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
            
        frame_count += 1
        display_image = image.copy()
        
        # Only process every few frames to maintain performance
        current_time = time.time()
        if current_time - last_check_time >= CHECK_INTERVAL:
            last_check_time = current_time
            
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) > 0:
                # Get face landmarks
                face_landmarks = face_recognition.face_landmarks(rgb_image)
                
                if face_landmarks:
                    # Get eye landmarks
                    left_eye = np.array(face_landmarks[0]['left_eye'])
                    right_eye = np.array(face_landmarks[0]['right_eye'])
                    
                    # Calculate eye aspect ratios
                    left_ear = calculate_eye_aspect_ratio(left_eye)
                    right_ear = calculate_eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Detect blink
                    if avg_ear < EYE_AR_THRESH:
                        blink_counter += 1
                        if blink_counter >= EYE_AR_CONSEC_FRAMES:
                            # Blink detected, save image
                            cv2.imwrite(f'{path}{Name}.png', image)
                            print(f"Blink detected! Image saved successfully!")
                            break
                    else:
                        blink_counter = 0
                    
                    # Draw eye aspect ratio
                    cv2.putText(display_image, f"EAR: {avg_ear:.2f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw face box
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_image, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        
        # Show instructions and status
        cv2.putText(display_image, "Blink to capture", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the image
        cv2.imshow('Blink Detection', display_image)
        
        # ESC to exit
        if cv2.waitKey(1) == 27:
            print("Capture cancelled")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    del(camera)

if __name__ == "__main__":
    Intial_data_capture()
