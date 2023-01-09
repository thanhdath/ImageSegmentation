import json
from warnings import resetwarnings
from flask import Flask, request, jsonify
import base64
import time
import string
import random
import os
import numpy as np
from flask_cors import CORS, cross_origin

from PIL import Image
from ade import CLASSES
from segmentation import get_largest_object_polygon, DeepLabModel

app = Flask(__name__)
cors = CORS(app)

TEMP_DIR = "./temp/"


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@app.route("/get_main_object", methods=["POST"])
def get_main_object():
    image = request.get_json()["image"]
    image_data = base64.b64decode(image)

    temp_filename = str(time.time()) + id_generator()
    temp_filepath = f"{TEMP_DIR}/{temp_filename}.png"

    with open(temp_filepath, "wb") as file:
        file.write(image_data)

    img = Image.open(temp_filepath)
    resized_im, seg_map = MODEL.run(img)

    filter_seg_map = np.zeros_like(seg_map, dtype=np.int32)
    for label in CONSIDER_CLASSES.keys():
        filter_seg_map[seg_map == ORI_CLASS2IDX[label]] = CONSIDER_CLASSES[label]

    box = get_largest_object_polygon(
        filter_seg_map, img.width, img.height, IDX2CONSIDER_CLASS
    )

    os.remove(temp_filepath)

    return jsonify(box)


if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)

    MODEL = DeepLabModel(
        "deeplabv3_mnv2_ade20k_train_2018_12_03/frozen_inference_graph.pb"
    )
    print("model loaded successfully!")

    ORI_CLASS2IDX = {k: i for i, k in enumerate(CLASSES)}

    CONSIDER_CLASSES = {
        "building, edifice": 1,
        "house": 1,
        "skyscraper": 1,
        "car, auto, automobile, machine, motorcar": 2,
        "truck, motortruck": 2,
    }  # class to our new label indices

    IDX2CONSIDER_CLASS = {1: "building", 2: "car+truck"}

    app.run()
