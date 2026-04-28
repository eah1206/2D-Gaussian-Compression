import cv2
import numpy as np
import os
import shutil
clip_name = "Patrick.mp4"
parent_path = 'Output_Data'

def save_data(frames: list, residuals: list):
    name = clip_name[:-4]
    print('Creating new folder...')
    new_path = name
    if os.path.exists(os.path.join(parent_path, new_path)):
        print('Folder already exists, clearing data...')
        shutil.rmtree(os.path.join(parent_path, new_path))

    frames_path = os.path.join(parent_path, os.path.join(new_path, 'frames'))
    residuals_path = os.path.join(parent_path, os.path.join(new_path, 'residuals'))
    os.makedirs(frames_path)
    os.makedirs(residuals_path)
    
    print("Saving frames...")
    for i in range(len(frames)):
        curr_frame = frames[i]
        bgr_data = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)
        image_path = os.path.join(frames_path, f"{name}_frame_{i:03}.png")
        cv2.imwrite(image_path, bgr_data)
    
    print("Saving residuals...")
    for i in range(len(residuals)):
        curr_frame = residuals[i]
        bgr_data = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)
        image_path = os.path.join(residuals_path, f"{name}_residual_{i:03}.png")
        cv2.imwrite(image_path, bgr_data)
    print("Saving Completed!")
    return

def display_images(images: list, name: str, fps: float):
    wait_time = int(1000/fps)
    for image in images:
        cv2.imshow(name, image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

clip = cv2.VideoCapture(clip_name)

clip_fps = clip.get(cv2.CAP_PROP_FPS)
frames = []

while True:
    end, frame = clip.read()
    
    if not end:
        break

    # Add each frame of the clip to an array
    frames.append(frame)

#display_images(frames, 'Clip Frame', clip_fps)

clip.release()

residuals = []

for i in range(1, len(frames)):
    frame1 = frames[i-1]
    frame2 = frames[i]

    residual = cv2.absdiff(frame1, frame2)
    residuals.append(residual)

#display_images(residuals, 'Residual', clip_fps)

save_data(frames, residuals)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()