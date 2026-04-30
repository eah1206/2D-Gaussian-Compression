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
        image_path = os.path.join(frames_path, f"{name}_frame_{i:03}.png")
        cv2.imwrite(image_path, curr_frame)
    
    print("Saving residuals...")
    for i in range(len(residuals)):
        curr_frame = residuals[i]
        image_path = os.path.join(residuals_path, f"{name}_residual_{i:03}.png")
        cv2.imwrite(image_path, curr_frame)
    print("Saving Completed!")
    return

def display_images(images: list, name: str, fps: float):
    wait_time = int(1000/fps)
    for image in images:
        cv2.imshow(name, image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def resize_image(frame):
    h, w = frame.shape[:2]
    if max(h, w) <= 256:
        return frame
    scale = 256 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


clip = cv2.VideoCapture(clip_name)

clip_fps = clip.get(cv2.CAP_PROP_FPS)
frames = []


while True:
    end, frame = clip.read()
    
    if not end:
        break
    
    frame = resize_image(frame)
    # Add each frame of the clip to an array
    frames.append(frame)

#display_images(frames, 'Clip Frame', clip_fps)

clip.release()

residuals = []

for i in range(1, len(frames)):
    frame1 = frames[i-1]
    frame2 = frames[i]

    #residual = cv2.absdiff(frame1, frame2)

    # -------------------------------------------------------------------------------------------------------------
    # Original implementation, slight ghosting when reassembled. Uncomment and comment other implementation to run
    # -------------------------------------------------------------------------------------------------------------
    diff = cv2.subtract(frame2.astype(np.int16), frame1.astype(np.int16))
    residual = np.clip(diff + 128, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------------------------------------------------------------------------
    # Other Implementaion. Residuals look weird, but reassembled seeingly losslessly. Uncomment and comment other implementation to run
    # ----------------------------------------------------------------------------------------------------------------------------------
    #diff = frame2.astype(np.int16) - frame1.astype(np.int16)  # range: [-255, 255]
    #residual = diff.astype(np.uint8)  # wraps mod 256, no shift needed
    #cv2.imwrite("residual.png", residual)  # PNG is still required!

    residuals.append(residual)

display_images(residuals, 'Residual', clip_fps)

save_data(frames, residuals)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()