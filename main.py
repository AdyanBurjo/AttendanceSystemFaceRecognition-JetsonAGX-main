
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
    
    args:
    name: str
    '''
    try:
        # Get current date for file access
        current_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%y_%m_%d")
        csv_path = f'Attendance_Entry/Attendance_{current_date}.csv'
        
        # If daily file doesn't exist, use backup file
        if not os.path.exists(csv_path):
            csv_path = "Attendance_Entry/Attendance_backup.csv"
        
        # Read existing entries
        nameList = []
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row:  # Check if row is not empty
                        nameList.append(row[0])
        except Exception as e:
            print(f"Error reading CSV: {e}")
            nameList = []  # Start fresh if file can't be read
        
        # Only add if name not already present
        if name not in nameList:
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str, date_str])
                print(f"Marked attendance for {name} at {time_str}")
            except Exception as e:
                print(f"Error marking attendance: {e}")
    
    except Exception as e:
        print(f"Unexpected error in markAttendance: {e}")
        # Create a new file if there's an error
        try:
            with open("Attendance_Entry/Attendance_backup.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime('%H:%M:%S'), 
                               datetime.now().strftime('%Y-%m-%d')])
        except Exception as backup_error:
            print(f"Critical error: Could not write to backup file: {backup_error}")

# Ensure Attendance_Entry directory exists
if not os.path.exists("Attendance_Entry"):
    os.makedirs("Attendance_Entry")

# Get current date for file naming
date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%y_%m_%d")
print(f"Current date: {date}")

# CSV file path
csv_path = f"Attendance_Entry/Attendance_{date}.csv"

# Create or check CSV file
if not os.path.exists(csv_path):
    try:
        with open(csv_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
        print(f"Created new attendance file: {csv_path}")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        csv_path = "Attendance_Entry/Attendance_backup.csv"
        with open(csv_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
        print(f"Created backup attendance file instead")

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
