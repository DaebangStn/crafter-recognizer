import os.path

import cv2
from glob import glob
import numpy as np


# window: numpy array of the window, must process to be RGB
def pattern_recognizer(window, reference_image_path):
    assert isinstance(window, np.ndarray), "window must be a numpy array"
    assert os.path.isfile(reference_image_path), f"ref_img_path: {reference_image_path} must be a valid file path"

    # Load source image and template image
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)

    # apply template matching
    res = cv2.matchTemplate(window, reference_image, cv2.TM_CCOEFF_NORMED)

    # set a threshold for the matching
    threshold = 0.8  # set your threshold here
    loc = np.where(res >= threshold)

    return list(zip(loc[0], loc[1]))


def recognized_tag_writer(window, recognizer_out):
    assert isinstance(window, np.ndarray), "window must be a numpy array"
    assert isinstance(recognizer_out, dict), "recognizer_out must be a dict"

    for tag, pt_list in recognizer_out.items():
        for pt in pt_list:
            # write a tag on the matching area
            cv2.putText(window, tag, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return window


def ambient_recognizer(window, reference_image_dir_path):
    assert isinstance(window, np.ndarray), "window must be a numpy array"
    assert os.path.isdir(reference_image_dir_path), f"ref_img_path: {reference_image_dir_path} must be a valid directory path"

    # convert BGR to RGB
    window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)

    # Load reference images
    reference_images = glob(os.path.join(reference_image_dir_path, '*.png'))

    recognizer_out_abs = {}

    for reference_image in reference_images:
        loc = pattern_recognizer(window, reference_image)
        image_name = os.path.basename(reference_image)  # xxxx.png
        image_name = os.path.splitext(image_name)[0]  # xxxx
        recognizer_out_abs[image_name] = loc
        print("loc: ", loc, "image_name: ", image_name)

    window = recognized_tag_writer(window, recognizer_out_abs)

    # change the recognizer_out to relative distance
    recognizer_out_rel = {}
    for tag, pt_list in recognizer_out_abs.items():
        dummy = []
        for pt in pt_list:
            dummy.append((pt[0] // 66, pt[1] // 66))
        recognizer_out_rel[tag] = np.unique(dummy, axis=0a).tolist()

    print("recognizer_out: ", recognizer_out_rel)

    # Display the image
    cv2.imshow("Detected Position", window)
    cv2.imwrite("Detected Position.jpg", window)


if __name__ == '__main__':
    pass
