import glob
import base64
import requests
import cv2
import os
import numpy as np

image_paths = glob.glob("data/images/*")

for img_path in image_paths:
    print(img_path)
    with open(img_path, "rb") as img:
        img_string = base64.b64encode(img.read()).decode("utf-8")

    response = requests.post(
        url="http://localhost:5000/get_main_object", json={"image": img_string}
    )

    print(response.json())

    if response.json() is not None:

        img = cv2.imread(img_path)
        img = cv2.polylines(
            img, [np.array(response.json()["points"])], True, (255, 0, 0), 2
        )

        os.makedirs("logs/", exist_ok=True)
        imname = img_path.split("/")[-1]
        cv2.imwrite(f"logs/{imname}", img)
