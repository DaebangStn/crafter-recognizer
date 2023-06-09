import os.path

import cv2
from pytesseract import pytesseract
from glob import glob
import numpy as np


# used opencv to match tile template. there is reference image in img folder
#   _window: numpy array of the window, must process to be RGB
def pattern_recognizer(_window, reference_image_path):
    assert isinstance(_window, np.ndarray), "window must be a numpy array"
    assert os.path.isfile(reference_image_path), f"ref_img_path: {reference_image_path} must be a valid file path"

    # Load source image and template image
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)

    # apply template matching
    res = cv2.matchTemplate(_window, reference_image, cv2.TM_CCOEFF_NORMED)

    # set a threshold for the matching
    threshold = 0.9  # set your threshold here
    loc = np.where(res >= threshold)

    return list(zip(loc[1], loc[0]))  # (x, y) format


# get the status of the player. inventory also included.
def status_recognizer(_window):
    assert isinstance(_window, np.ndarray), "window must be a numpy array"

    # crop the window to get the status, must be fine-tuned
    window_status = {
        "health": _window[470:540, 20:85],
        "hunger": _window[470:540, 91:145],
        "water": _window[470:540, 145:210],
        "energy": _window[470:540, 210:285],

        "sapling": _window[470:540, 285:350],
        "wood": _window[470:540, 350:415],
        "stone": _window[470:540, 415:480],
        "coal": _window[470:540, 480:545],
        "iron": _window[470:540, 540:600],

        "wood_pickaxe": _window[535:600, 90:145],
        "stone_pickaxe": _window[535:600, 145:220],
        "iron_pickaxe": _window[535:600, 210:285],

        "wood_sword": _window[535:600, 285:350],
        "stone_sword": _window[535:600, 350:415],
        "iron_sword": _window[535:600, 415:490],
    }

    thresh_white = 250

    for key in window_status.keys():
        window_status[key] = cv2.cvtColor(window_status[key], cv2.COLOR_RGB2GRAY)
        window_status[key] = cv2.threshold(window_status[key], thresh_white, 255, cv2.THRESH_BINARY)[1]

    # if you want to see the image, uncomment the following code
#    key = "coal"
#    cv2.imshow(key, window_read[key])

    for key in window_status.keys():
        text = pytesseract.image_to_string(window_status[key], config='--psm 13 digits')

        # remove all non-digit characters (the first character is always a digit)
        # there could be empty string
        if len(text) > 0:
            text = text[0]
        elif text == '':
            # number 4 is similar to empty string. so we need to check the mean brightness of the image
            if np.mean(window_status[key]) > 30:
                text = int(4)
            # if the mean brightness is low, it is empty (0)
            else:
                text = int(0)
        try:
            window_status[key] = int(text)
        except ValueError:
            window_status[key] = int(0)
            print("ValueError: ", key, text, type(text))

    return window_status


# write the recognized tag on the window.
# you need absolute position of the tag (not relative position)
# recognizer_out: {tag: list of absolute position}
def recognized_tag_writer(_window, recognizer_out):
    assert isinstance(_window, np.ndarray), "window must be a numpy array"
    assert isinstance(recognizer_out, dict), "recognizer_out must be a dict"

    for tag, pt_list in recognizer_out.items():
        if isinstance(pt_list, list):
            for pt in pt_list:
                # write a tag on the matching area
                cv2.putText(_window, tag, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return _window


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

    ambient_rel = convert_absolute_ambient_to_relative(ambient_abs)
    print("ambient relative position: ", ambient_rel)

    isNight = night_detector(ambient_rel)
    ambient_rel["isNight"] = isNight

    window_status = status_recognizer(window)
    print("window status: ", window_status)

    window = recognized_tag_writer(window, ambient_abs)
    # Display the image
    cv2.imshow("Detected Position", window)
    cv2.imwrite("Detected Position.jpg", window)


# divide the coordinate by 66 to get the tile position
# subtract (4, 3) to get the relative position
# make the coordinate unique
def convert_absolute_ambient_to_relative(ambient_abs):
    ambient_rel = {}
    for tag, pt_list_abs in ambient_abs.items():
        pt_list_rel = []
        for pt in pt_list_abs:
            pt_list_rel.append((pt[0] // 66, pt[1] // 66))
        pt_list_rel = np.unique(pt_list_rel, axis=0).tolist()
        ambient_rel[tag] = []
        for pt in pt_list_rel:
            ambient_rel[tag].append([a - b for a, b in zip(pt, [4, 3])])
    return ambient_rel


# return True if it is night
# if recognition rate is low, it is night
def night_detector(ambient_rel):
    # sum all length of ambient_rel.values()
    count = sum(len(v) for v in ambient_rel.values())
    print("count: ", count)
    if count < 20:
        return True
    else:
        return False



if __name__ == '__main__':
    window = cv2.imread("Detected Position.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("window", window)
    status_recognizer(window)
    # wait forever
    cv2.waitKey(0)
    cv2.destroyAllWindows()

