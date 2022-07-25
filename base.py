import csv
import mediapipe as mp
import numpy as np 
import cv2 as cv
import time



def media_pipe():
    mp_drawing= mp.solutions.drawing_utils # drawing helper
    mp_pose= mp.solutions.pose # mediapipe model
    return mp_drawing,mp_pose

def Resize(frame,scale):
            height=int(frame.shape[0]* scale)
            width=int(frame.shape[1]*scale)
            dimensions=(width,height)

            return cv.resize(frame,dimensions,interpolation=cv.INTER_CUBIC)

def pose_model_base(mp_drawing,mp_pose):
    video=cv.VideoCapture(0)
    with mp_pose.Pose(model_complexity=2,min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        time1= 0

 
        while video.isOpened():
            _, frame= video.read()
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

        # Pose landmarks drawing
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(178, 247, 31),thickness=2,circle_radius=1),mp_drawing.DrawingSpec(color=(129, 102, 108),thickness=1))

        
            cv.imshow('Raw Webcam Feed', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv.destroyAllWindows()

    return results


def create_file(results):
    num_objects=len(results.pose_landmarks.landmark)
    yoga=['class']
    yoga += list((np.array([[f'x_{i}',f'y_{i}',f'z_{i}',f'v_{i}'] for i in range(1,num_objects+1)]).flatten()))
    with open('landmarks.csv','w',newline='') as f:
        file=csv.writer(f,delimiter=',')
        file.writerow(yoga)

if __name__=='__main__':
    mp_drawing,mp_pose= media_pipe()
    results=pose_model_base(mp_drawing,mp_pose)
    create_file(results)
   



