import os
import cv2
import enum
import json
import shutil
import numpy as np
import tkinter as tk
from send2trash import send2trash

class SizeUnit(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
        
class DuplicateViewer(object):
    def __init__(self, duplicates_json):
        with open(duplicates_json) as json_file:
            print("Reading duplicate data from json file...")
            self.data = json.load(json_file)
            print("Finished reading json into dictionary")
        self.window = "window-name"
        self.running = True
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        duplicate_count = 0
        for d in self.data.values():
            if len(d) > 1:
                works = True
                for i in d:
                    if not os.path.exists(i):
                        works = False
                        break
                if works:
                    duplicate_count += 1
        print("Found " + str(duplicate_count) + " files with duplicates")
        for d in self.data.values():
            if self.running and len(d) > 1:
                self.display_duplicates(d)

    def display_duplicates(self, files):
        image_width = int(self.screen_width / len(files))
        images = []
        flags = {}
        for index, path in enumerate(files):
            try:
                image = cv2.imread(path)
                _ = image.shape # check for exceptions
            except:
                try:
                    capture = cv2.VideoCapture(path)
                    success, image = capture.read()
                    capture.release()
                    assert success
                except:
                    print("Failed to process file format for " + path)
            try:
                image = cv2.resize(image, (image_width, self.screen_height), interpolation = cv2.INTER_AREA)
                self.add_centered_text(image, DuplicateViewer.get_file_size_text(path), image_width, 50, 0.8)
                self.add_centered_text(image, path, image_width, 70, 1 / len(files))
                images.append(image)
                flags[index] = (path, True)
            except:
                pass
        flag_list = list(flags.values())
        if len(flag_list) > 1:
            original_images = np.copy(images)
            first_path, first_flag = flag_list[0]
            self.add_border(images, files, flags, first_path, 0)
            horizontal_image = cv2.hconcat(images)
            cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(self.window, 0, 0)
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.window, horizontal_image)
            while cv2.getWindowProperty(self.window, 0) >= 0:
                k = cv2.waitKey(33)
                if k == 27: # Esc key to stop
                    self.running = False
                    break
                elif k == -1: # -1 returned if no key is pressed
                    continue
                elif k == ord('s'): # Skip duplicates if 's' is pressed
                    break
                elif k == ord(' '): # Delete all non-flagged duplicates if space is pressed
                    kept_images = 0
                    for index, file in flags.items():
                        path, flag = file
                        if not flag:
                            kept_images += 1
                    if kept_images <= 0: # Don't delete if no image is kept
                        break
                    else:
                        for index, file in flags.items():
                            path, flag = file
                            if flag:
                                if os.path.exists(path):
                                    try:
                                        send2trash(os.path.abspath(path))
                                        print("Deleting: " + path)
                                    except:
                                        shutil.move(os.path.abspath(path), 'D:/Media/test')
                                        print("Moving: " + path + " to duplicate folder")
                        print("---")
                    break
                else:
                    key = chr(k)
                    try:
                        index = int(key) - 1
                        if index >= 0 and index < len(files):
                            path, flag = flags[index]
                            if flag:
                                self.add_border(images, files, flags, path, index)
                            else:
                                images[index] = original_images[index]
                                flags[index] = (path, True)
                            new_horizontal_image = cv2.hconcat(images)
                            cv2.imshow(self.window, new_horizontal_image)
                    except:
                        pass

    @staticmethod
    def convert_unit(size, unit):
        if unit == SizeUnit.KB:
            return size / 1024, "KB"
        elif unit == SizeUnit.MB:
            return size / (1024 * 1024), "MB"
        elif unit == SizeUnit.GB:
            return size / (1024 * 1024 * 1024), "GB"
        else:
            return size, "BYTES"

    @staticmethod
    def get_file_size_text(file_path, unit = SizeUnit.MB):
        size = os.path.getsize(file_path)
        converted_size, suffix = DuplicateViewer.convert_unit(size, unit)
        return str(round(converted_size, 2)) + " " + suffix

    def add_centered_text(self, image, text, image_width, y, font_scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        background_color = (255, 255, 255)
        font_color = (0, 0, 0)
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        x = int(image_width / 2 - text_width / 2)
        cv2.rectangle(image, (x, y), (x + text_width, y + text_height), background_color, -1)
        y = int(y + text_height + font_scale - 1)
        cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

    def add_border(self, images, files, flags, path, index):
        assert index < len(images)
        image = images[index]
        shape = image.shape
        h, w, channels = shape
        background = np.zeros(shape, dtype=np.uint8)
        bc = (0, 255, 0) # border color
        bt = 20 # border thickess
        cv2.rectangle(background, (0, 0), (w, h), bc, -1)
        # fill part of background with original image
        background[bt:h - bt, bt:w - bt] = image[bt:h - bt, bt:w - bt]
        flags[index] = (path, False)
        images[index] = background



DuplicateViewer('C:/Dev/duplicate_finder/test_file.json')

print("Exiting program")