import cv2
from segmentation import DeepLabModel


def get_largest_object(image):
    im_width = image.shape[1]
    im_height = image.shape[0]

    img = loadmat(file)["segment"]
    print(img.shape, image.shape)

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
