import cv2
import mediapipe as mp
import time
mpHand=mp.solutions.hands
hand=mpHand.Hands()
draw=mp.solutions.drawing_utils
cTime=0
pTime=0
video=cv2.VideoCapture(1)
while True:
    s,image=video.read()
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=hand.process(rgb)
    if result.multi_hand_landmarks:
        for point in result.multi_hand_landmarks:
            for id,lm in enumerate(point.landmark):
                h,w,c=image.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
            draw.draw_landmarks(image,point,mpHand.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow('Output',image)
    if(cv2.waitKey(1) & 0xff==ord('z')):
        break







