
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import traceback
import pytz
import csv


def identifyEncodings(images, classNames):
    '''
    Encoding is Recognition and comparing particular face in database or stored folder

    args:
    images: list of images
    classNames: list of image names
    '''
    
    encodeList = []
    for img, name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encode = encodings[0]
            encodeList.append(encode)
        else:
            print(f"Warning: No face detected in image for {name}")
            # Remove the corresponding name from classNames
            classNames.remove(name)
            continue
    return encodeList

def markAttendance(name):
    '''
    This function handles attendance marking in CSV file
    Records every detection without limitations
    
    args:
    name: str
    '''
    try:
        # Ensure the directory exists
        os.makedirs("Attendance_Entry", exist_ok=True)
        
        # Use a fixed filename that doesn't depend on date/time
        attendance_file = "Attendance_Entry/Attendance_Log.csv"
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date"])
        
        # Record the attendance (always add new entry)
        try:
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, time_str, date_str])
            print(f"Logged detection of {name} at {time_str}")
            
        except Exception as e:
            print(f"Error logging attendance: {e}")
            # If there's an error with the main file, use a backup
            backup_file = "Attendance_Entry/Attendance_Backup.csv"
            with open(backup_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # If file is empty, write header
                    writer.writerow(["Name", "Time", "Date"])
                writer.writerow([name, time_str, date_str])
            print(f"Logged to backup file instead")
    
    except Exception as e:
        print(f"Critical error in markAttendance: {e}")
        print("Continuing with face detection...")

# Ensure Attendance_Entry directory exists
os.makedirs("Attendance_Entry", exist_ok=True)

# Initialize the attendance log file if it doesn't exist
attendance_file = "Attendance_Entry/Attendance_Log.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time", "Date"])
    print("Initialized new attendance log file")

#Preprocessing the data 

path = 'Attendance_data'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
# split the data vk.png to vk
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Encoding of input image data
encodeListKnown = identifyEncodings(images, classNames)
print('Encoding Complete')
print(f'Successfully encoded {len(encodeListKnown)} faces')


#Camera capture 
cap = cv2.VideoCapture(0)  # Use default camera on Windows

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #Face recognition using dlib - limit to 1 face
    facesCurFrame = face_recognition.face_locations(imgS)
    
    # Only process the first face found
    if len(facesCurFrame) > 0:
        # Take only the first face
        facesCurFrame = [facesCurFrame[0]]
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        # If multiple faces are detected, draw a warning
        if len(face_recognition.face_locations(imgS)) > 1:
            cv2.putText(img, "Please show only one face", (10, 30), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    else:
        encodesCurFrame = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)  # Even stricter tolerance
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Only proceed if we have any matches and the best match is very confident
        if True in matches:
            matchIndex = np.argmin(faceDis)
            # Much stricter threshold for acceptance
            if faceDis[matchIndex] < 0.4:  # Very strict matching threshold
                confidence = 1 - faceDis[matchIndex]
                # Only accept if confidence is very high
                if confidence > 0.6:  # Requires 60% confidence
                    name = classNames[matchIndex].upper()
                    print(f"Detected: {name} (Confidence: {confidence:.2%})")
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

    cv2.imshow('Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
