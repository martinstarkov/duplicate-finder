import json
import cv2
import enum
import os
import numpy as np
from PIL import Image
import sys
import tkinter as tk
from pathlib import Path, WindowsPath
from send2trash import send2trash
try:
    import pyheif # only on UNIX
except ImportError:
    pass

class SizeUnit(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4

class Picture(object):
    def __init__(self, path, border, image_width):
        self.image_width = image_width
        self.path = path
        self.image = DuplicateViewer.find_image(str(self.path))
        self.refresh()
        self.border = border

    def refresh(self):
        self.original_image = np.copy(self.image)

    def reset(self):
        self.image = np.copy(self.original_image)

    def resize(self, width, height):
        try:
            self.image = cv2.resize(self.image, (width, height), interpolation = cv2.INTER_AREA)
            self.refresh()
            return False
        except:
            return True

    def add_border(self):
        shape = self.image.shape
        h, w, channels = shape
        background = np.zeros(shape, dtype=np.uint8)
        bc = (0, 255, 0) # border color
        bt = 20 # border thickess
        cv2.rectangle(background, (0, 0), (w, h), bc, -1)
        # Fill part of background with original image
        background[bt:h - bt, bt:w - bt] = self.image[bt:h - bt, bt:w - bt]
        self.image = background

    def add_centered_text(self, text, y, font_scale, pic_count):
        font = cv2.FONT_HERSHEY_SIMPLEX
        background_color = (255, 255, 255)
        font_color = (0, 0, 0)
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        x = int(self.image_width / pic_count / 2 - text_width / pic_count)
        cv2.rectangle(self.image, (x, y), (x + text_width, y + text_height), background_color, -1)
        y = int(y + text_height + font_scale - 1)
        cv2.putText(self.image, text, (x, y), font, font_scale, font_color, thickness)
        self.refresh()

class DuplicateViewer(object):
    def __init__(self, duplicate_json_file):
        with open(duplicate_json_file) as file:
            print("Reading duplicate data from json file...")
            self.duplicate_data = json.load(file)
            print("Finished reading json into dictionary")
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.image_width = int(self.screen_width / 2)
        self.window = "Duplicate Viewer"
        self.display_duplicates()

    @staticmethod
    def find_image(file: str):
        image = None
        path = Path(file)
        suffix = path.suffix.lower()
        try:
            readable = suffix == '.jpg' or suffix == '.png' or suffix == '.bmp' or suffix == '.jpeg'
            if not readable:
                raise Exception
            image = cv2.imread(file)
        except:
            try:
                if 'pyheif' in sys.modules:
                    heif_file = pyheif.read(file)
                    image = Image.frombytes(
                        heif_file.mode, 
                        heif_file.size, 
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    image = np.array(image)
                elif suffix == '.heif' or suffix == '.heic':
                    pass
                else:
                    raise Exception
            except:
                try:
                    if suffix == '.mov' or suffix == '.mp4' or suffix == '.avi':
                        capture = cv2.VideoCapture(file)
                        success, image = capture.read()
                        capture.release()
                    else:
                        raise Exception
                except:
                    image = None
                    print("Could not recognize file format for: " + file)
        return image

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

    def display_duplicates(self):
        self.running = True
        counter = 0
        print(len(self.duplicate_data[0]))
        for i, hash in enumerate(self.duplicate_data[0]):
            if self.running:
                paths = [hash[0], hash[1]]
                similarity = hash[2]
                pictures = []
                path_count = len(paths)
                counter += 1
                print(counter)
                if path_count > 1 and path_count < 10: #and float(similarity) < 7.0:
                    for index, path in enumerate(paths):
                        if os.path.exists(path):
                            pic = Picture(path, False, self.screen_width)
                            if pic is not None:
                                pictures.append(pic)
                    pic_count = len(pictures)
                    if pic_count > 1 and int(self.screen_width / pic_count) != 0:
                        pictures[0].border = True
                        to_skip = False
                        for pic in pictures:
                            to_skip = pic.resize(int(self.screen_width / pic_count), self.screen_height)
                            if to_skip:
                                break
                            pic.add_centered_text(DuplicateViewer.get_file_size_text(pic.path), 50, 0.8, pic_count)
                            pic.add_centered_text(Path(pic.path).suffix, 70, 1 / 2, pic_count)
                            pic.add_centered_text(str(similarity), 90, 1 / 2, pic_count)
                        for pic in pictures:
                            if pic is None:
                                to_skip = True
                                break
                        if to_skip:
                            continue
                        else:
                            if os.path.isfile(pictures[0].path) and os.path.isfile(pictures[1].path):
                                cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
                                cv2.moveWindow(self.window, 0, 0)
                                cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                self.update_display(pictures)
                                while cv2.getWindowProperty(self.window, 0) >= 0:
                                    k = cv2.waitKey(33)
                                    if k == 27: # Esc key to stop
                                        self.running = False
                                        break
                                    elif k == -1: # -1 returned if no key is pressed
                                        continue
                                    elif k == ord('s'): # Skip duplicate if 's' is pressed
                                        break
                                    elif k == ord('1') or k == ord('2') or k == ord('3') or k == ord('4') or k == ord('5') or k == ord('6') or k == ord('7') or k == ord('8') or k == ord('9'):
                                        key = 0
                                        if k == ord('1'):
                                            key = 1
                                        elif k == ord('2'):
                                            key = 2
                                        elif k == ord('3'):
                                            key = 3
                                        elif k == ord('4'):
                                            key = 4
                                        elif k == ord('5'):
                                            key = 5
                                        elif k == ord('6'):
                                            key = 6
                                        elif k == ord('7'):
                                            key = 7
                                        elif k == ord('8'):
                                            key = 8
                                        elif k == ord('9'):
                                            key = 9
                                        if key <= pic_count:
                                            pictures[key - 1].border = not pictures[key - 1].border
                                            self.update_display(pictures)
                                    elif k == ord(' '):
                                        for picture in pictures:
                                            if not picture.border:
                                                if os.path.exists(picture.path):
                                                    try:
                                                        send2trash(os.path.abspath(picture.path))
                                                        print("Deleting: " + picture.path)
                                                    except:
                                                        print("Failed to delete: " + picture.path)
                                                        #shutil.move(os.path.abspath(picture.path), 'D:/Media/test')
                                                        #print("Moving: " + picture.path + " to duplicate folder")
                                        print("---")
                                        break
                else:
                    pass
                        # Notify the user that an image cannot be displayed
                        #print("Skipping: [" + picture1.path + ", " + picture2.path + "]")

    def update_display(self, pictures):
        images = []
        for picture in pictures:
            if picture.border:
                picture.add_border()
            else:
                picture.reset()
            images.append(picture.image)
        cv2.imshow(self.window, cv2.hconcat(images))

def flip_json(path_from, path_to):
    duplicates = {}
    with open(path_from) as file:
        duplicates = json.load(file)
    dictionary = {}
    for path in duplicates:
        if os.path.exists(path):
            hash = duplicates[path]
            if hash in dictionary:
                dictionary[hash].append(path)
            else:
                dictionary[hash] = [path]
    json_object = json.dumps(dictionary, indent=4)
    with open(path_to, "w") as outfile:
        outfile.write(json_object)

def eliminate_singles(path_from, path_to):
    dictionary = {}
    with open(path_from) as file:
        dictionary = json.load(file)
    pop_hashes = []
    for hash in dictionary:
        path_array = dictionary[hash]
        if len(path_array) < 2:
            pop_hashes.append(hash)
    for hash in pop_hashes:
        dictionary.pop(hash)
    json_object = json.dumps(dictionary, indent=4)
    with open(path_to, "w") as outfile:
        outfile.write(json_object)

#flip_json("../hashes.json", "../hashes_correct.json")
#eliminate_singles("../hashes_correct.json", "../hashes_dupes.json")
#viewer = DuplicateViewer('../hashes_dupes.json')

viewer = DuplicateViewer('../output_duplicates.json')