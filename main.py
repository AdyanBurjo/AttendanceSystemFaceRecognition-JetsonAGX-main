
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
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
        small_frame = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
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
        # Ensure the directory exists
        os.makedirs("Attendance_Entry", exist_ok=True)
        
        # Use a fixed filename for today's date
        now = datetime.now()
        current_date = now.strftime("%y_%m_%d")
        attendance_file = f'Attendance_Entry/Attendance_{current_date}.csv'
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date"])
        
        # Record the attendance
        time_str = now.strftime('%H:%M:%S')
        date_str = now.strftime('%Y-%m-%d')
        
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_str, date_str])
        print(f"Logged attendance for {name} at {time_str}")
    
    except Exception as e:
        print(f"Error marking attendance: {e}")
        # If there's an error, try using a backup file
        try:
            backup_file = "Attendance_Entry/Attendance_Backup.csv"
            with open(backup_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # If file is empty, write header
                    writer.writerow(["Name", "Time", "Date"])
                writer.writerow([name, now.strftime('%H:%M:%S'), now.strftime('%Y-%m-%d')])
            print("Logged to backup file instead")
        except Exception as backup_error:
            print(f"Failed to write to backup file: {backup_error}")

# Ensure Attendance_Entry directory exists
os.makedirs("Attendance_Entry", exist_ok=True)

# Create today's attendance file
current_date = datetime.now().strftime("%y_%m_%d")
attendance_file = f"Attendance_Entry/Attendance_{current_date}.csv"

# Create file with headers if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time", "Date"])
    print(f"Created new attendance file for today: {attendance_file}")
else:
    print(f"Using today's attendance file: {attendance_file}")

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
    small_frame = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Face recognition using dlib
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

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
