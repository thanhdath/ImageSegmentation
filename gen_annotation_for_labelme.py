from scipy.io import loadmat
import glob
import os
from ade import CLASSES
import cv2
import numpy as np
from get_scores_road import combine_images
import base64
import dicttoxml

files = sorted(glob.glob("output/segment_img/*"))

CLASS2IDX = {val: i for i, val in enumerate(CLASSES)}
print("Number of classes", len(CLASS2IDX))
IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}

CLASS2COLOR = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 3,
    13: 3,
    14: 4,
    15: 4,
    16: 4,
    17: 4,
    18: 4,
    19: 4,
}

GROUP2COLOR = {"human": 1, "nature": 2, "building": 3, "vehicle": 4}

CONSIDER_CLASSES = {
    "person, individual, someone, somebody, mortal, soul": 1,
    "grass": 2,
    "tree": 3,
    "mountain, mount": 4,
    "plant, flora, plant life": 3,
    "water": 5,
    "sea": 6,
    "rock, stone": 7,
    "sand": 8,
    "river": 5,
    "flower": 3,
    "palm, palm tree": 9,
    "land, ground, soil": 10,
    "waterfall, falls": 11,
    "building, edifice": 12,
    "house": 12,
    "skyscraper": 13,
    "car, auto, automobile, machine, motorcar": 14,
    "railing, rail": 15,
    "boat": 16,
    "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle": 17,
    "truck, motortruck": 14,
    "ship": 16,
    "minibike, motorbike": 18,
    "bicycle, bike, wheel, cycle": 19,
}  # class to our new label indices
CLASS2IDX = {k: i for i, k in enumerate(CLASSES)}

OUTPUT_FOR_LABELING = "output/for_labeling/mat"
OUTPUT_FOR_LABELING_IMG = "output/for_labeling/img"

if not os.path.isdir(OUTPUT_FOR_LABELING):
    os.makedirs(OUTPUT_FOR_LABELING)
if not os.path.isdir(OUTPUT_FOR_LABELING_IMG):
    os.makedirs(OUTPUT_FOR_LABELING_IMG)

seg2nclasses = {}

## just count class in consider class
for ifile, file in enumerate(files):
    if ifile % 100 == 0:
        print("Processed ", ifile)
    seg_map = loadmat(file)["segment"]
    filter_seg_map = seg_map
    # print(seg_map)
    # filter_seg_map = np.zeros_like(seg_map, dtype=np.int32)
    # for label in CONSIDER_CLASSES.keys():
    #     filter_seg_map[seg_map == CLASS2IDX[label]] = CONSIDER_CLASSES[label]
    # chỉ xem xét 1 vài class
    n_classes = len(set(filter_seg_map.flatten()))
    seg2nclasses[file] = n_classes
    print(seg2nclasses)
    # print(file, n_classes)

files_label = sorted(seg2nclasses.items(), key=lambda x: x[1], reverse=True)[:300]
print(files_label)

from shutil import copyfile

for file, nclasses in files_label:
    filename = file.split("/")[-1]
    copyfile(file, OUTPUT_FOR_LABELING + "/" + filename)
    filename = filename.replace(".mat", ".jpeg")
    img = cv2.imread("data/images/" + filename)
    # img = cv2.resize(img, (513, 513))
    cv2.imwrite(OUTPUT_FOR_LABELING_IMG + "/" + filename, img)

print("Gen annotation")

import glob
import cv2
import imutils
import os
import json

CLASSES = {
    "human": 1,
    "grass": 2,
    "tree+plant+flower": 3,
    "mountain": 4,
    "river+water": 5,
    "sea": 6,
    "rock, stone": 7,
    "sand": 8,
    "palm tree": 9,
    "soil": 10,
    "waterfall": 11,
    "building+house": 12,
    "skyscraper": 13,
    "car+truck": 14,
    "railing, rail": 15,
    "boat+ship": 16,
    "bus": 17,
    "motorbike": 18,
    "bicycle": 19,
}
IDX2CLASS = {v: k for k, v in CLASSES.items()}
files = sorted(glob.glob(OUTPUT_FOR_LABELING + "/*"))

for ifile, file in enumerate(files):
    if ifile % 100 == 0:
        print("Processed ", ifile)

    imagePath = file.split("/")[-1].replace(".mat", ".jpeg")
    ori_img = cv2.imread(OUTPUT_FOR_LABELING_IMG + "/" + imagePath)
    im_width = ori_img.shape[1]
    im_height = ori_img.shape[0]

    img = loadmat(file)["segment"]
    print(img.shape, ori_img.shape)

    filter_seg_map = np.zeros_like(img, dtype=np.int32)
    for label in ["building+house", "car+truck"]:
        filter_seg_map[img == CLASSES[label]] = CLASSES[label]
    img = filter_seg_map

    vals = set(img.flatten())

    anno = {
        "version": "3.16.1",
        "flags": {},
        "shapes": [],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": imagePath,
        "imageData": base64.b64encode(
            open(OUTPUT_FOR_LABELING_IMG + "/" + imagePath, "rb").read()
        ).decode(),
        "imageHeight": im_height,
        "imageWidth": im_width,
    }

    for val in vals:
        if val != 0 and val != 15:
            mask = np.zeros_like(img, dtype=np.uint8)
            # smooth image
            mask[img == val] = 255
            kernel = np.ones((7, 7), np.float32) / 49
            mask = cv2.filter2D(mask, -1, kernel)
            mask[mask >= 127] = 255
            mask[mask < 127] = 0
            mask[mask == 255] = 1
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # end smooth image

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [x for x in contours if cv2.contourArea(x) > 20]
            for contour in contours:
                # smooth contour
                contour = contour.reshape(-1, 2)
                new_contour = [contour[0]]
                for i in range(1, len(contour)):
                    prev_pt = new_contour[-1]
                    dist = np.sqrt(np.sum((contour[i] - prev_pt) ** 2))
                    if dist > 10:
                        new_contour.append(contour[i])
                if len(new_contour) >= 3:
                    contour = new_contour
                box = {
                    "label": IDX2CLASS[val],
                    "line_color": None,
                    "fill_color": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "points": [
                        [
                            int(x) * im_width / mask.shape[1],
                            int(y) * im_height / mask.shape[0],
                        ]
                        for x, y in contour
                    ],
                }
                anno["shapes"].append(box)
    with open(
        OUTPUT_FOR_LABELING_IMG + "/" + imagePath.replace(".jpeg", ".json"), "w+"
    ) as fp:
        fp.write(json.dumps(anno, indent=4, sort_keys=True))
