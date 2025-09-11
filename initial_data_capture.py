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

def detect_face_orientation(landmarks):
    """
    Detect face orientation using facial landmarks
    Returns: left, right, or center
    """
    # Get key facial points
    nose_tip = np.array(landmarks['nose_tip'][0])
    left_eye_pts = np.array(landmarks['left_eye'])
    right_eye_pts = np.array(landmarks['right_eye'])
    
    # Calculate eye centers
    left_eye = np.mean(left_eye_pts, axis=0)
    right_eye = np.mean(right_eye_pts, axis=0)
    eyes_center = (left_eye + right_eye) / 2
    
    # Calculate eye widths (use for asymmetry detection)
    left_eye_width = np.linalg.norm(left_eye_pts[0] - left_eye_pts[3])
    right_eye_width = np.linalg.norm(right_eye_pts[0] - right_eye_pts[3])
    eye_width_ratio = right_eye_width / left_eye_width if left_eye_width > 0 else 1.0
    
    # Calculate horizontal difference relative to eye width
    eye_distance = abs(right_eye[0] - left_eye[0])
    nose_offset = nose_tip[0] - eyes_center[0]
    
    # Normalize the offset by eye distance
    normalized_offset = nose_offset / (eye_distance * 0.5)
    
    # Thresholds for movement detection
    CENTER_THRESHOLD = 0.3       # Smaller range for center
    TURN_THRESHOLD = 0.6        # Lower threshold to detect turns easier
    
    # Use both normalized offset and eye width ratio for detection
    if abs(normalized_offset) <= CENTER_THRESHOLD and 0.8 < eye_width_ratio < 1.2:
        return "center"
    elif normalized_offset < -TURN_THRESHOLD or eye_width_ratio > 1.3:  # Left turn makes left eye appear larger
        return "right"
    elif normalized_offset > TURN_THRESHOLD or eye_width_ratio < 0.5:   # Right turn makes right eye appear larger
        return "left"
    return "center"  # Default to center if not clearly left or right

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
    
    # Constants for detection
    EYE_BLINK_THRESHOLD = 0.2  # Stricter threshold for blink detection
    ORIENTATION_HOLD_TIME = 2.0  # Time to hold each position (2 seconds)
    
    # Movement sequence states
    movement_sequence = ["center", "right", "blink"]
    current_movement = 0
    movement_start_time = 0
    orientation_confirmed = False
    
    # Blink detection variables
    blink_counter = 0
    consecutive_blink_frames = 0
    is_eyes_closed = False
    last_blink_time = 0
    required_blinks = 3
    
    print("\nInstructions:")
    print("Please follow these steps in order:")
    print("1. Look at CENTER for 2 seconds")
    print("2. Turn RIGHT for 2 seconds")
    print("3. Look at CENTER and blink 3 times")
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
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            # Scale landmarks back to original image size
            scaled_landmarks = {}
            for feature, points in landmarks.items():
                scaled_points = []
                for point in points:
                    scaled_points.append([point[0] * 4, point[1] * 4])
                scaled_landmarks[feature] = scaled_points
            
            # Draw facial landmarks for visualization with thicker lines
            for feature, points in scaled_landmarks.items():
                points = np.array(points)
                cv2.polylines(display_image, [points], True, (0, 255, 0), 2)
                # For key points like eyes, nose, and mouth, add dots
                if feature in ['left_eye', 'right_eye', 'nose_tip', 'top_lip', 'bottom_lip']:
                    for point in points:
                        cv2.circle(display_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            
            # Get current orientation
            orientation = detect_face_orientation(landmarks)
            
            # Get image dimensions for text placement
            height, width = display_image.shape[:2]
            
            # Show current face orientation (debug info)
            orientation_status = f"Current Position: {orientation.upper()}"
            cv2.putText(display_image, orientation_status,
                       (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            
            # Handle movement sequence
            required_orientation = movement_sequence[current_movement]
            if required_orientation != "blink":  # Handle face movements
                
                if orientation == required_orientation and not orientation_confirmed:
                    if movement_start_time == 0:
                        movement_start_time = current_time
                    elif (current_time - movement_start_time) >= ORIENTATION_HOLD_TIME:
                        orientation_confirmed = True
                        print(f"{required_orientation.upper()} position confirmed!")
                else:
                    movement_start_time = 0
                
                if orientation_confirmed:
                    current_movement += 1
                    orientation_confirmed = False
                    movement_start_time = 0
                
                # Get image dimensions
                height, width = display_image.shape[:2]
                center_x = width // 2
                center_y = height // 2

                # Get instruction based on current required orientation
                required_orientation = movement_sequence[current_movement]
                if required_orientation == "center":
                    instruction_text = "LOOK AT CENTER"
                elif required_orientation == "right":
                    instruction_text = "TURN RIGHT"
                else:  # blink phase
                    instruction_text = f"BLINK {blink_counter}/{required_blinks}"
                
                # Add remaining time if holding position
                if movement_start_time > 0 and required_orientation != "blink":
                    remaining_time = ORIENTATION_HOLD_TIME - (current_time - movement_start_time)
                    if remaining_time > 0:
                        instruction_text += f" ({remaining_time:.1f}s)"

                # Draw step and instruction in top left
                progress = f"Step {current_movement + 1} of {len(movement_sequence)}"
                cv2.putText(display_image, progress,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                cv2.putText(display_image, instruction_text,
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
            
            else:  # Handle blinking phase
                # Get eye landmarks
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                
                # Calculate eye aspect ratios
                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if orientation == "center":
                    # Check for blink
                    if avg_ear < EYE_BLINK_THRESHOLD:
                        consecutive_blink_frames += 1
                        if consecutive_blink_frames >= 2:  # Need 2 consecutive frames of closed eyes
                            if not is_eyes_closed and (current_time - last_blink_time) > 1.0:  # At least 1 second between blinks
                                blink_counter += 1
                                last_blink_time = current_time
                                is_eyes_closed = True
                    else:
                        is_eyes_closed = False
                        consecutive_blink_frames = 0
                    
                    # Display blink status in top left
                    progress = f"Step {current_movement + 1} of {len(movement_sequence)}"
                    cv2.putText(display_image, progress,
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 255, 255), 2)
                    cv2.putText(display_image, "Look at CENTER and BLINK",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Blinks: {blink_counter}/{required_blinks}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                else:
                    progress = f"Step {current_movement + 1} of {len(movement_sequence)}"
                    cv2.putText(display_image, progress,
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 255, 255), 2)
                    cv2.putText(display_image, "Please look at CENTER",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2)
                
                # Check if all conditions are met
                if blink_counter >= required_blinks:
                    # Add delay after last blink
                    if (current_time - last_blink_time) < 1.0:  # Wait for 1 second
                        cv2.putText(display_image, "Get ready for capture...", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_image, "CAPTURING!", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Save image
                        cv2.imwrite(f'{path}{Name}'+'.png', image)
                        print(f"Image saved successfully with face detected!")
                        break
        
        # Show the image
        cv2.imshow('Capturing', display_image)
        
        # Check for ESC key
        if cv2.waitKey(1) == 27:  # ESC
            print("Capture cancelled")
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    
Intial_data_capture()
