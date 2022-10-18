import cv2
import mediapipe as mp
import time

pTime,cTime=0,0
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture("poseVideo/2.mp4")
while True:
    success,img=cap.read()
    resize = cv2.resize(img, (700, 700))
    imgRB=cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(resize,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=resize.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(resize,(cx,cy),5,(255,0,255),cv2.FILLED)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(resize,str(int(fps)),(10,78),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    
    cv2.imshow("Image",resize)
    key=cv2.waitKey(10)
    #ascii for q and Q to quit
    if key==81 or key==113:
        break
cap.release()
