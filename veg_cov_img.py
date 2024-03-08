import glob
import os
import cv2
import numpy as np

# Params
exgi_threshold = 12

# Process photos
extensions = ["jpg", "jpeg", "png"]

if not os.path.exists('results'):
    os.makedirs('results')

for extension in extensions:
    for img_path in glob.glob(f'./img/**/*.{extension}', recursive=True):
        handle = cv2.imread(img_path)

        matrix_exgi = 2 * handle[:, :, 1].astype(np.float32) \
            - handle[:, :, 0].astype(np.float32) \
            - handle[:, :, 2].astype(np.float32) \
            - exgi_threshold
        matrix_boolean = np.clip(matrix_exgi, 0, 1).astype(np.uint8)
        veg_cover = cv2.threshold(matrix_boolean, 0.5, 255, cv2.THRESH_BINARY)[1]
        
        processed_img_path = os.path.join(
            './results/', os.path.basename(img_path).split('.')[0] + '_cover.' + extension
        )

        cv2.imwrite(processed_img_path, veg_cover)
        
        perc = np.sum(veg_cover) / (veg_cover.shape[0] * veg_cover.shape[1]) / 255
        print(perc)
