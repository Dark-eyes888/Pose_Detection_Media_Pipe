# importing libraries
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import time

from base import media_pipe



# Get Camera feed

def pose_detection(model):
    mp_drawing,mp_pose=media_pipe()
    
    with open(model,'rb') as f:
        model_gbc= pickle.load(f)
    video=cv.VideoCapture(0)

    with mp_pose.Pose(model_complexity=2,min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        time1= 0
            
        while video.isOpened():
            _, frame= video.read()
            frame.flags.writeable=False
            img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                
            img=cv.flip(img,1)
            # make detections
            result=pose.process(img)

            time2= time.time()

            if (time2 - time1)>0:
                fps=round(1.0/(time2-time1),1)

                cv.putText(img,f'FPS : {fps}',(10,30),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
          

            time1=time2
            img.flags.writeable=True
            img =cv.cvtColor(img,cv.COLOR_RGB2BGR)

            # Pose Detections
            Class_landmarks= result.pose_landmarks.landmark
            landmarks= list(np.array([[i.x,i.y,i.x,i.visibility] for i in Class_landmarks]).flatten())


            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(178, 247, 31),thickness=2,circle_radius=1),mp_drawing.DrawingSpec(color=(129, 102, 108),thickness=1))
                
            data=pd.DataFrame([landmarks])
    
            Face_body_class= pd.Series(model_gbc.predict(data))
            texts=Face_body_class.map({0:'Normal',1:'Downdog',2:'Goddess',3:'Warrior2'})[0]
            
            prob_class=model_gbc.predict_proba(data)[0]
            print(texts,prob_class)

            cv.putText(img,texts,(200,30),cv.FONT_HERSHEY_PLAIN, 2,(255, 0, 255),2,cv.LINE_AA)
        
            cv.imshow('Raw Webcam Feed', img)



            if cv.waitKey(1) & 0xFF == ord('q'):
                    
                break

    video.release()
    cv.destroyAllWindows()  


if __name__=="__main__":
    pose_detection('xgb.pkl')
