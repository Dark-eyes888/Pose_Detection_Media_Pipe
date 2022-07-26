
## POSE DETECTION USING MEDIA PIPE

It was really fun working on this project. I am sharing some of my insights on Pose detection. I am starting with 3 yoga poses - Downdog, Goddesss, and Warrior. This repo is still under development. I will be adding face mesh and hand landmarks also.


This repository consists of pose detection method using media pipe for body landmarks detection and drawing. 
The code is divided into 5 parts: base.py, img_df.py, video_df.py,  model.py, pose_detection.py

The Base.py file contains basic media pipe landmark detection and gathering x,y,z,visibility coordinates of all 33 landmarks into a csv.
 
we can use video,images or web-cam to create dataset for different poses to classify.
I used images and web-cam to gather data and appended all the coordinates into csv. Seperate codes for images and video/webcam are added in this repository. So we can use one or combination of all for better dataset.


 The total columns created and appended are(4*33 +1 ) -> one is added for classes.

For pose classification, I used xgboost classifier with a training accuracy of 1.0 and validation accuraccy of 0.99. 

The final part, the 2 seperate models were working together- blazepoze for landmark detection and xgboost for classification.


# The best part, we can train and tune our model for any kind of body movement. It is not limited to specific task or movement. 



