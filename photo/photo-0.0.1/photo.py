# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 14:00:52 2017

@author: Yanchen
@version: python2.7
"""

import cv2
import os
import datetime

# Initialize the camera
cap = cv2.VideoCapture(0);

while(True):
	# Capture frame-by-frame
    ret, frame = cap.read();

    # Our operations on the frame come here
    display = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR);

    # Display the resulting frame
    cv2.imshow('frame', display);
    
    # Wait for 1 seconds and exit when press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    
    # Wait for 60 mile seconds and take photos when press p    
    if cv2.waitKey(60) & 0xFF == ord('p'):
        # Get the current time
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S');
        # Close current window
        cv2.destroyAllWindows();
        # If filepath not exist, create it
        if (not os.path.exists(os.getcwd() + '\\photo')):
            os.makedirs(os.getcwd() + '\\photo');
        # Save photo to filepath
        cv2.imwrite(os.getcwd() + '\\photo\\' + filename + '.jpg', frame[:,:,0]);
        continue;

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
