import cv2
import numpy as np

# Params
exgi_threshold = 12

# Open the video file
video_path = 'vid/input.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('vid/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    matrix_exgi = 2 * frame[:, :, 1].astype(np.float32) \
        - frame[:, :, 0].astype(np.float32) \
        - frame[:, :, 2].astype(np.float32) \
        - exgi_threshold
    matrix_boolean = np.clip(matrix_exgi, 0, 1).astype(np.uint8)
    veg_cover = cv2.threshold(matrix_boolean, 0.5, 255, cv2.THRESH_BINARY)[1]

    # Write processed frame to output video
    out.write(veg_cover)

    # Display the processed frame
    cv2.imshow('Processed Frame', veg_cover)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release video capture and writer
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
