import cv2
import numpy as np
import os

clip_name = "Patrick.mp4"


def save_data(frames: list, residuals: list):


    for frame in frames:
        break
    return


clip = cv2.VideoCapture(clip_name)

frames = []

while True:
    end, frame = clip.read()
    
    if not end:
        break

    # Add each frame of the clip to an array
    frames.append(frame)

for frame in frames:
    # Process 'frame' (it is already a NumPy array for OpenCV)
    cv2.imshow('Clip Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print(type(frames[0]))
clip.release()
cv2.destroyAllWindows()

residuals = []

for i in range(1, len(frames)):
    frame1 = frames[i-1]
    frame2 = frames[i]

    residual = cv2.absdiff(frame1, frame2)
    residuals.append(residual)

for residual in residuals:
    cv2.imshow('Residual', residual)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print(type(residuals[0]))
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()