import cv2
import numpy as np
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def distance(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def get_ear(landmark, check):
    #left
    up = distance(landmark[37],landmark[41]) + distance(landmark[38],landmark[40])
    down = distance(landmark[36],landmark[39])
    left = up/(2.0*down)
    if check == 1:
        return left
    #right
    up = distance(landmark[43],landmark[47]) + distance(landmark[44],landmark[46])
    down = distance(landmark[42],landmark[45])
    right = up/(2.0*down)
    return right

def avg_ear(landmark):
    left = get_ear(landmark,1)
    right = get_ear(landmark,0)
    return (left + right)/2
    

def blinked(landmark, check):
	ratio = get_ear(landmark,check)

	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks,1)
        right_blink = blinked(landmarks,0)
        EAR = avg_ear(landmarks)
        
        if(left_blink==0 or right_blink==0):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>9):
        		status="SLEEPING !!!"
        		color = (0,0,255)

        elif(left_blink==1 or right_blink==1):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>9):
        		status="Drowsy"
        		color = (255,0,0)

        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>9):
        		status="Active"
        		color = (0,255,0)
        	
        cv2.putText(frame, status, (face_frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for n in range(0, 68):
        	(x,y) = landmarks[n]
        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
