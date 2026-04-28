import cv2

gif = cv2.VideoCapture('The Car Crash - 1080p H.264.mp4')

frames = []

while True:
    end, frame = gif.read()
    
    if not end:
        break

    # Add each frame of the gif to an array
    frames.append(frame)

    # Process 'frame' (it is already a NumPy array for OpenCV)
    cv2.imshow('GIF Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

gif.release()
cv2.destroyAllWindows()

for i in range(1, len(frames)):
    frame1 = frames[i-1]
    frame2 = frames[i]

    residual = cv2.absdiff(frame1, frame2)

    cv2.imshow('Residual', residual)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()