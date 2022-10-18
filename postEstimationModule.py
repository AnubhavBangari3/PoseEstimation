import cv2
import mediapipe as mp
import time
#https://google.github.io/mediapipe/solutions/pose#python-solution-api
class PoseDetector():
    def __init__(self,mode=False,upBody=False,smooth=True,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findPose(self,resize,draw=True):
        imgRB=cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRB)
        

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(resize,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return resize
    
    def getPos(self,resize,draw=True):
        lmLists=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=resize.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmLists.append([id,cx,cy])
                if draw:
                    cv2.circle(resize,(cx,cy),5,(255,0,255),cv2.FILLED)
        return lmLists
    

def main():
    pTime,cTime=0,0
    cap=cv2.VideoCapture("poseVideo/2.mp4")
    detector=PoseDetector()
    while True:
        success,img=cap.read()
        resize = cv2.resize(img, (700, 700))
        resize=detector.findPose(resize)
        lmLists=detector.getPos(resize)
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
    


if __name__ == '__main__':
    main()