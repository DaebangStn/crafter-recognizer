import os.path

import cv2
from pytesseract import pytesseract
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
    threshold = 0.9  # set your threshold here
    loc = np.where(res >= threshold)

    return list(zip(loc[1], loc[0]))  # (x, y) format


def status_recognizer(window):
    assert isinstance(window, np.ndarray), "window must be a numpy array"

    window_status = {
        "health": window[470:540, 20:85],
        "hunger": window[470:540, 91:145],
        "water": window[470:540, 145:210],
        "energy": window[470:540, 210:285],

        "sapling": window[470:540, 285:350],
        "wood": window[470:540, 350:415],
        "stone": window[470:540, 415:480],
        "coal": window[470:540, 480:545],
        "iron": window[470:540, 540:600],

        "wood_pickaxe": window[535:600, 90:145],
        "stone_pickaxe": window[535:600, 145:220],
        "iron_pickaxe": window[535:600, 210:285],

        "wood_sword": window[535:600, 285:350],
        "stone_sword": window[535:600, 350:415],
        "iron_sword": window[535:600, 415:490],
    }

    thresh_white = 250

    for key in window_status.keys():
        window_status[key] = cv2.cvtColor(window_status[key], cv2.COLOR_RGB2GRAY)
        window_status[key] = cv2.threshold(window_status[key], thresh_white, 255, cv2.THRESH_BINARY)[1]

#    key = "coal"
#    cv2.imshow(key, window_read[key])

    for key in window_status.keys():
        text = pytesseract.image_to_string(window_status[key], config='--psm 13 digits')
        if len(text) > 0:
            text = text[0]

        if text == '':
            if np.mean(window_status[key]) > 30:
                text = int(4)
            else:
                text = int(0)
        try:
            window_status[key] = int(text)
        except ValueError:
            window_status[key] = int(0)
            print("ValueError: ", key, text, type(text))

    return window_status


def recognized_tag_writer(window, recognizer_out):
    assert isinstance(window, np.ndarray), "window must be a numpy array"
    assert isinstance(recognizer_out, dict), "recognizer_out must be a dict"

    for tag, pt_list in recognizer_out.items():
        if isinstance(pt_list, list):
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

    ambient_abs = {}

    for reference_image in reference_images:
        loc = pattern_recognizer(window, reference_image)
        image_name = os.path.basename(reference_image)  # xx-yy.png
        image_name = os.path.splitext(image_name)[0]  # xx-yy
        image_name = image_name.split('-')[0]  # xx

        if image_name not in ambient_abs.keys():
            if len(loc) == 0:
                ambient_abs[image_name] = []
            else:
                ambient_abs[image_name] = loc
        elif isinstance(ambient_abs[image_name], list):
            if len(loc) > 0:
                ambient_abs[image_name] += loc


    # change the recognizer_out to relative distance
    ambient_rel = {}
    for tag, pt_list_abs in ambient_abs.items():
        pt_list_rel = []
        for pt in pt_list_abs:
            pt_list_rel.append((pt[0] // 66, pt[1] // 66))
        pt_list_rel = np.unique(pt_list_rel, axis=0).tolist()
        ambient_rel[tag] = []
        for pt in pt_list_rel:
            ambient_rel[tag].append([a - b for a, b in zip(pt, [4, 3])])

    print("ambient relative position: ", ambient_rel)

    window_status = status_recognizer(window)
    print("window status: ", window_status)

    window = recognized_tag_writer(window, ambient_abs)

    # Display the image
    cv2.imshow("Detected Position", window)
    cv2.imwrite("Detected Position.jpg", window)


if __name__ == '__main__':
    window = cv2.imread("Detected Position.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("window", window)
    status_recognizer(window)
    # wait forever
    cv2.waitKey(0)
    cv2.destroyAllWindows()

