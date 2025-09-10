
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
    global current_attendance_file  # Use the global file path
    
    try:
        # Ensure the directory exists
        os.makedirs("Attendance_Entry", exist_ok=True)
        
        # Use the global attendance file that was created at startup
        if not os.path.exists(current_attendance_file):
            # Create new file with headers if it doesn't exist
            with open(current_attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date"])
        
        # Read existing entries
        nameList = []
        try:
            with open(current_attendance_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row:  # Check if row is not empty
                        nameList.append(row[0])
        except Exception as e:
            print(f"Error reading CSV: {e}")
            nameList = []
        
        # Only add if name not already present in today's list
        if name not in nameList:
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            
            try:
                with open(current_attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str, date_str])
                print(f"Marked attendance for {name} at {time_str}")
            except Exception as e:
                print(f"Error marking attendance: {e}")
                # If there's an error, try creating a new file
                with open(current_attendance_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Name", "Time", "Date"])
                    writer.writerow([name, time_str, date_str])
    
    except Exception as e:
        print(f"Unexpected error in markAttendance: {e}")
        # If all else fails, use a backup file
        backup_file = "Attendance_Entry/Attendance_backup.csv"
        try:
            with open(backup_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # If file is empty, write header
                    writer.writerow(["Name", "Time", "Date"])
                writer.writerow([name, datetime.now().strftime('%H:%M:%S'), 
                               datetime.now().strftime('%Y-%m-%d')])
            print(f"Attendance marked in backup file: {backup_file}")
        except Exception as backup_error:
            print(f"Critical error: Could not write to backup file: {backup_error}")

# Initialize global variables
current_attendance_file = None

# Ensure Attendance_Entry directory exists
os.makedirs("Attendance_Entry", exist_ok=True)

# Set up the attendance file for today
try:
    # Get current date for file naming (only date, not time)
    current_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%y_%m_%d")
    current_attendance_file = f"Attendance_Entry/Attendance_{current_date}.csv"
    
    # Create new file with headers if it doesn't exist
    if not os.path.exists(current_attendance_file):
        with open(current_attendance_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
        print(f"Created new attendance file: {current_attendance_file}")
    else:
        print(f"Using existing attendance file: {current_attendance_file}")

except Exception as e:
    print(f"Error setting up attendance file: {e}")
    current_attendance_file = "Attendance_Entry/Attendance_backup.csv"
    # Create backup file if needed
    if not os.path.exists(current_attendance_file):
        with open(current_attendance_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
    print(f"Using backup attendance file: {current_attendance_file}")

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
