import os
import cv2
import enum
import numpy as np
import tkinter as tk
from send2trash import send2trash

class File(object):
    def __init__(self, path, thumbnail, flag = True):
        self.path = path
        self.thumbnail = thumbnail
        self.flag = flag

class DuplicateFinder(object):
    def __init__(self, directories):
        self.duplicates = self.merge_duplicates(directories)

    @staticmethod
    # Joins two dictionaries
    def merge_dictionaries(dictionary1, dictionary2):
        for key in dictionary2.keys():
            if key in dictionary1:
                dictionary1[key] = dictionary1[key] + dictionary2[key]
            else:
                dictionary1[key] = dictionary2[key]

    def merge_duplicates(self, directories):
        duplicates = {}
        for directory in directories:
            if os.path.exists(directory):
                DuplicateFinder.merge_dictionaries(duplicates, self.find_duplicates(directory))
            else:
                print('%s is not a valid path, please verify' % directory)
                return {}
        return duplicates
        
    def find_duplicates(self, parent_directory):
        duplicates = {}
        for directory, _, files in os.walk(parent_directory):
            print('Scanning %s...' % directory)
            for file in files:
                path = os.path.join(directory, file)
                if not os.path.exists(path):
                    continue
                try:
                    image = cv2.imread(path)
                    _ = image.shape # check for exceptions
                except:
                    try:
                        capture = cv2.VideoCapture(path)
                        success, image = capture.read()
                        assert success
                    except:
                        print("Failed to process file format for " + path)
                try:
                    _ = image.shape
                    file_hash = cv2.img_hash.BlockMeanHash_create().compute(image).tobytes()
                    if file_hash in duplicates:
                        duplicates[file_hash].append(File(path, image))
                    else:
                        duplicates[file_hash] = [File(path, image)]
                except:
                    pass
        return duplicates

class SizeUnit(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
        
class DuplicateViewer(object):
    def __init__(self, duplicates):
        self.duplicates = duplicates
        self.window = "window-name"
        self.running = True
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        for d in self.duplicates.values():
            if self.running and len(d) > 1:
                self.display_duplicates(d)

    def display_duplicates(self, files):
        image_width = int(self.screen_width / len(files))
        images = []
        for index, file in enumerate(files):
            image = cv2.resize(file.thumbnail, (image_width, self.screen_height), interpolation = cv2.INTER_AREA)
            self.add_centered_text(image, DuplicateViewer.get_file_size_text(file.path), image_width, 50, 0.8)
            self.add_centered_text(image, file.path, image_width, 70, 1 / len(files))
            images.append(image)
        original_images = np.copy(images)
        self.add_border(images, files, 0)
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
                for file in files:
                    if not file.flag:
                        kept_images += 1
                if kept_images <= 0: # Don't delete if no image is kept
                    break
                else:
                    for file in files:
                        if file.flag:
                            print("Deleting: " + file.path)
                            send2trash(os.path.abspath(file.path))
                    print("---")
                break
            else:
                key = chr(k)
                index = int(key) - 1
                if index >= 0 and index < len(files):
                    if files[index].flag:
                        self.add_border(images, files, index)
                    else:
                        images[index] = original_images[index]
                        files[index].flag = True
                    new_horizontal_image = cv2.hconcat(images)
                    cv2.imshow(self.window, new_horizontal_image)

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

    def add_border(self, images, files, index):
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
        files[index].flag = False
        images[index] = background

DuplicateViewer(DuplicateFinder(['path/here']).duplicates)

print("Exiting program")