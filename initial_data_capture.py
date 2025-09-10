import cv2
import face_recognition
import os

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
