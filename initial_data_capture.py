import cv2
import face_recognition
import time

def Intial_data_capture(camera_id=None):
    """
    Capture a face image using a simple countdown timer when face is detected.
    
    args:
    camera_id : int
    """
    path = "Attendance_data/"
    if camera_id == None:
        camera_id = 1  # Use default camera on Windows
    
    Name = input("Please Enter your name: ")

    camera = cv2.VideoCapture(camera_id)
    
    face_detected = False
    countdown_started = False
    countdown = 5
    last_time = time.time()
    
    while True:
        return_value, image = camera.read()
        if not return_value:
            print("Failed to capture image")
            break
        
        # Create a copy for drawing
        display_image = image.copy()
        
        # Only check for face every few frames to improve performance
        if not face_detected or time.time() - last_time >= 0.5:  # Check every 0.5 seconds
            # Convert to RGB for face_recognition (only when needed)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            last_time = time.time()
            
            if len(face_locations) > 0:
                face_detected = True
                # Draw rectangle around face
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_image, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
            else:
                face_detected = False
                countdown_started = False
                countdown = 5
        
        # If face is detected, start/continue countdown
        if face_detected:
            if not countdown_started:
                countdown_started = True
                countdown_start = time.time()
            
            # Update countdown
            elapsed = time.time() - countdown_start
            countdown = max(0, 5 - int(elapsed))
            
            # Show countdown
            cv2.putText(display_image, f"Capturing in: {countdown}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Capture image when countdown reaches 0
            if countdown == 0:
                cv2.imwrite(f'{path}{Name}.png', image)
                print(f"Image saved successfully!")
                break
        else:
            # Show instruction
            cv2.putText(display_image, "Position your face in the camera", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Face Capture', display_image)
        
        # ESC to exit
        if cv2.waitKey(1) == 27:
            print("Capture cancelled")
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    del(camera)

if __name__ == "__main__":
    Intial_data_capture()
