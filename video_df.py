# This file describes the custom data generation using web-cam or any video file. 
# we will use web cam for this.

from base import media_pipe,Resize
import numpy as np 
import cv2 as cv
import csv
import time



def video_cam(pose_name):
    
    mp_drawing,mp_pose=media_pipe()
    
    class_name= pose_name
    cap=cv.VideoCapture(0)
    with mp_pose.Pose(model_complexity=2,static_image_mode=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
       time1= 0
       
       
       while cap.isOpened():
            _, frame= cap.read()
            frame.flags.writeable=False
            re_img=Resize(frame,1.25)

        # change color feed
        
            img=cv.cvtColor(re_img,cv.COLOR_BGR2RGB)
        
            img=cv.flip(img,1)

        # make detections
            results=pose.process(img)

            time2= time.time()

            if (time2 - time1)>0:
                
                fps=round(1.0/(time2-time1),1)

                cv.putText(img,f'FPS : {fps}',(10,30),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
        

            time1=time2
            img.flags.writeable=True
            img =cv.cvtColor(img,cv.COLOR_RGB2BGR)

            # Drawing landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(178, 247, 31),thickness=2,circle_radius=1),mp_drawing.DrawingSpec(color=(129, 102, 108),thickness=1))
            
            Class_landmarks= results.pose_landmarks.landmark
            landmarks= list(np.array([[i.x,i.y,i.x,i.visibility] for i in Class_landmarks]).flatten())
            landmarks.insert(0,class_name)

            with open('landmarks.csv','a',newline='') as f:
                File= csv.writer(f, delimiter=',')
                File.writerow(landmarks)
                
            
            cv.imshow('Raw Feed', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()          
    cv.destroyAllWindows()

if __name__=='__main__':
    video_cam('normal')
