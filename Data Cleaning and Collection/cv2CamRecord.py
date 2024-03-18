#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
We would like to thank Nick Renotte for his youtube tutorial on how to collect images
using our personal webcams

We followed instructions from his youtube video and well as his github, which
we have pasted below:
https://www.youtube.com/watch?v=pDXdlXlaCco
https://github.com/nicknochnack/RealTimeObjectDetection.git

We made some edits to the code to make it label the 
correct letters we will be using for the model.
"""


# In[1]:


import cv2
import os
import time
import uuid

IMAGES_PATH = 'CollectedImages'

# Labels of letters we will be using for our model to generate "hello world"
labels = ['h', 'e', 'l', 'o', 'w', 'r', 'd'] 

# collecting 100 images per letter
number_imgs = 100

# Loop through each of the labels in the labels array
for label in labels:
    # Create a directory for each label
    label_directory = os.path.join(IMAGES_PATH, label)
    os.makedirs(label_directory, exist_ok=True)
    
    # Beginning video capture (should see a popup on computer screen)
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    
    # Allows for 5 seconds to change positions when capturing images
    time.sleep(5)
    
    # Looking at whether the the webcame capture works
    if not cap.isOpened():
        print("Error: Could not open webcam")
        break

    # Taking the amount of images we set above, which is 100 images per letter
    for imgnum in range(number_imgs):
        # Setting up the frame for our images
        ret, frame = cap.read()
        
        if ret:
            # Creating the name of our image file
            imgname = os.path.join(label_directory, label + '_' + str(uuid.uuid1()) + '.jpg')
            
            # Adding the images into our directory
            cv2.imwrite(imgname, frame)
            
            # Displaying the iamge on our screen to see what we have taken
            cv2.imshow('frame', frame)
            
            # Pauses to get into another pose
            time.sleep(2)

        # Looking at whether the user wants to quit taking images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finishes the capture of image for a letter
    cap.release()

# Closes the camera popup window
cv2.destroyAllWindows()

