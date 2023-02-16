from flask import Flask, request, jsonify
import time
import string
import random
import os
import numpy as np
from flask_cors import CORS, cross_origin
import cv2

from PIL import Image
from ade import CLASSES
from segmentation import get_top_k_objects_polygon, DeepLabModel

app = Flask(__name__)
cors = CORS(app)

TEMP_DIR = "./temp/"


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@app.route("/get_main_object", methods=["POST"])
def get_main_object():
    file = request.files["image"]
    max_n_objects = int(request.form.get("max_n_objects", 1))
    x1 = float(request.form.get("x1"))
    print(x1)
    y1 = float(request.form.get("y1"))
    print(y1)
    x2 = float(request.form.get("x2"))
    print(x2)
    y2 = float(request.form.get("y2"))
    print(y2)
    print(file)

    original_filename = file.filename
    original_ext = original_filename.split(".")[-1]

    temp_filename = str(time.time()) + id_generator()
    temp_filepath = f"{TEMP_DIR}/{temp_filename}.{original_ext}"

    file.save(temp_filepath)

    img = Image.open(temp_filepath)
    print("Size: ", img.size)
    img = img.crop((x1, y1, x2, y2))
    _, seg_map = MODEL.run(img)

    filter_seg_map = np.zeros_like(seg_map, dtype=np.int32)
    for label in CONSIDER_CLASSES.keys():
        filter_seg_map[seg_map == ORI_CLASS2IDX[label]] = CONSIDER_CLASSES[label]
    boxes = get_top_k_objects_polygon(
        filter_seg_map,
        x1,
        y1,
        img.width,
        img.height,
        IDX2CONSIDER_CLASS,
        max_n_objects=max_n_objects,
    )

    if len(boxes) > 0:
        img_cv = np.asarray(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        for box in boxes:
            img_cv = cv2.polylines(
                img_cv, [np.array(box["points"])], True, (255, 0, 0), thickness=2
            )
            cv2.imwrite(f"{TEMP_DIR}/{temp_filename}-box.png", img_cv)
    else:
        print("len(points)==0")
    os.remove(temp_filepath)

    return jsonify(boxes)


if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)

    MODEL = DeepLabModel(
        "deeplabv3_xception_ade20k_train/frozen_inference_graph.pb"
        # "deeplabv3_mnv2_ade20k_train_2018_12_03/frozen_inference_graph.pb"
    )
    print("model loaded successfully!")

    ORI_CLASS2IDX = {k: i for i, k in enumerate(CLASSES)}

    CONSIDER_CLASSES = {
        "building, edifice": 1,
        "house": 1,
        "skyscraper": 1,
        "car, auto, automobile, machine, motorcar": 2,
        "truck, motortruck": 2,
        "airplane, aeroplane, plane": 3,
    }  # class to our new label indices

    IDX2CONSIDER_CLASS = {1: "building", 2: "car+truck", 3: "plane"}

    app.run()
