# import libraries
import os
import csv
from base import media_pipe
import numpy as np 
import cv2 as cv



def img_values():
    mp_drawing,mp_pose=media_pipe()
    
    pose_name=os.listdir(r'C:\Users\adm88\ML_deployment\Object_detection\yoga_pose_media_pipe\TEST')

    with mp_pose.Pose(model_complexity=2,static_image_mode=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

        for i in pose_name:
            class_name=i
            img_dir_path=r'C:\Users\adm88\ML_deployment\Object_detection\yoga_pose_media_pipe\TEST\{0}'.format(i)
            for j in os.listdir(img_dir_path):
                pic=cv.imread(img_dir_path+"\\"+j)

            # detecting poses
            
                pic=cv.cvtColor(pic,cv.COLOR_BGR2RGB)
                pic=cv.resize(pic,(540,540),cv.INTER_LINEAR)
        
            
            # make detections
                result=pose.process(pic)
            # change color feed
        
                pic=cv.cvtColor(pic,cv.COLOR_RGB2BGR)
            

            # Pose Detections
            
                mp_drawing.draw_landmarks(pic, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(178, 247, 31),thickness=2,circle_radius=1),mp_drawing.DrawingSpec(color=(129, 102, 108),thickness=1))
            
                Class_landmarks= result.pose_landmarks.landmark
                landmarks= list(np.array([[i.x,i.y,i.x,i.visibility] for i in Class_landmarks]).flatten())
                landmarks.insert(0,class_name)

                with open('landmarks.csv','a',newline='') as f:
                    File= csv.writer(f, delimiter=',')
                    File.writerow(landmarks)
                
            
                    cv.imshow('Raw Feed', pic) #can comment out this line 

                    cv.waitKey(1) 
            
            cv.destroyAllWindows() 
    
if __name__== '__main__':
    img_values()