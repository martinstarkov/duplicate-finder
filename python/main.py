import os
import cv2
import sys
import json
import enum
import dhash
import numpy as np
from send2trash import send2trash
from pathlib import Path, WindowsPath
from PIL import Image, ImageOps

try:
    import pyheif # Only on UNIX.
except ImportError:
    pass

class SizeUnit(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
    
class Picture(object):
    def __init__(self, path: str, border: bool, image_width: int):
        self.image_width = image_width
        self.path = path
        self.image = ImageProcessor.find_image(path)
        self.refresh()
        self.border = border

    def refresh(self):
        self.original_image = np.copy(self.image)

    def reset(self):
        self.image = np.copy(self.original_image)

    def resize(self, width, height):
        self.image = cv2.resize(self.image, (width, height), interpolation = cv2.INTER_AREA)
        self.refresh()

    def setup(self, screen_height, similarity):
        self.resize(self.image_width, screen_height)
        self.add_centered_text("Similarity: " + str(similarity), 20, 0.8)
        self.add_centered_text(DuplicateViewer.get_file_size_text(self.path), 50, 0.8)
        self.add_centered_text(self.path, 70, 1 / 2)

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

    def add_centered_text(self, text, y, font_scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        background_color = (255, 255, 255)
        font_color = (0, 0, 0)
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        x = int(self.image_width / 2 - text_width / 2)
        cv2.rectangle(self.image, (x, y), (x + text_width, y + text_height), background_color, -1)
        y = int(y + text_height + font_scale - 1)
        cv2.putText(self.image, text, (x, y), font, font_scale, font_color, thickness)
        self.refresh()

class DuplicatePair(dict):
    def __init__(self, hash1, path1: Path, hash2, path2: Path, similarity: int):
        self.hash1 = hash1
        self.path1 = path1
        self.hash2 = hash2
        self.path2 = path2
        self.similarity = similarity
        dict.__init__(self, paths=[str(self.path1), str(self.path2)], similarity=self.similarity)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.hash1 == other.hash1 and self.hash2 == other.hash2) or \
                   (self.hash1 == other.hash2 and self.hash2 == other.hash1)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class ImageProcessor(object):
    @staticmethod
    def is_image(extension: str):
        return extension == '.jpg' or \
               extension == '.png' or \
               extension == '.bmp' or \
               extension == '.jpeg'

    @staticmethod
    def is_video(extension: str):
        return extension == '.mov' or \
               extension == '.mp4' or \
               extension == '.wmv' or \
               extension == '.gif' or \
               extension == '.avi'
    
    @staticmethod
    def is_heic(extension: str):
        return extension == '.heic' or \
               extension == '.heif'

    @staticmethod
    def find_image(file: str):
        image = None
        path = Path(file)
        if path.is_file():
            extension = path.suffix.lower()
            if ImageProcessor.is_image(extension):
                image = cv2.imread(file)
            elif ImageProcessor.is_video(extension):
                capture = cv2.VideoCapture(file)
                success, image = capture.read()
                capture.release()
            elif 'pyheif' in sys.modules and ImageProcessor.is_heic(extension):
                heic = pyheif.read(file)
                image = Image.frombytes(heic.mode, heic.size, heic.data, "raw", heic.mode, heic.stride,)
                image = np.array(image)
            else:
                pass
                #print("[Invalid file extension]: " + file)
        return image

class DuplicateFinder(object):
    # Similarity threshold indicates how many of the 128-bits in the image hash
    # can be different before the images are no longer considered duplicates. 
    # e.g. 0/128 different -> identical images, 10/128 different -> likely different
    def __init__(self, directories: list, destination: str, similarity_threshold: int):
        paths = DuplicateFinder.gather_paths(directories)
        hashes = DuplicateFinder.calculate_hashes(paths)
        duplicates = DuplicateFinder.find_duplicates(hashes, similarity_threshold)
        DuplicateFinder.save_to_file(destination, duplicates)
    
    @staticmethod
    def gather_paths(directories: list):
        paths = []
        for directory in directories:
            directory = Path(directory)
            if directory.is_dir():
                print("[Processing directory]: " + str(directory))
                # Recurse through all subdirectories and return all filepaths inside of them.
                for path in directory.rglob("*"):
                    paths.append(path)
                print("[Processed directory]")
            else:
                print("[Invalid directory]: " + str(directory))
        return paths

    @staticmethod
    def calculate_hashes(paths: list):
        print("[Calculating hashes...]")
        hashes = []
        for index, path in enumerate(paths):
            image = ImageProcessor.find_image(str(path.resolve()))
            if not image is None:
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grey, (8, 8))
                hash = dhash.dhash_int(Image.fromarray(resized), size=8)
                hashes.append((hash, path))
                #print("Found hash for " + str(path.resolve()))
            percent_complete = int(index / len(paths) * 100)
            if percent_complete % 10 == 0:
                print("[" + str(percent_complete) + "% of hashes calculated]")
        print("[Calculated all hashes]")
        return hashes

    @staticmethod
    def find_duplicates(hashes: list, threshold: int):
        print("[Finding duplicates...]")
        duplicates = []
        index = 0
        for hash1, path1 in hashes:
            for hash2, path2 in hashes:
                if path1 != path2:
                    hamming_distance = dhash.get_num_bits_different(hash1, hash2)
                    if hamming_distance <= threshold:
                        similarity = int((1 - hamming_distance / (threshold + 1)) * 100)
                        pair = DuplicatePair(hash1, path1, hash2, path2, similarity)
                        if not DuplicateFinder.identical_pair_exists(pair, duplicates):
                            duplicates.append(pair)
            percent_complete = int(index / len(hashes) * 100)
            if percent_complete % 10 == 0:
                print("[" + str(percent_complete) + "% of duplicates found]")
            index += 1
            # file = str(path1)
            # if hash1 in duplicates:
            #     duplicates[hash1].append(file)
            # else:
            #     duplicates[hash1] = [file]
        print("[Found all duplicates]")
        return duplicates

    @staticmethod
    def identical_pair_exists(pair: DuplicatePair, duplicates: list):
        for duplicate_pair in duplicates:
            if duplicate_pair == pair:
                return True
        return False

    @staticmethod
    def save_to_file(destination: str, duplicates: list):
        file = open(destination, "w")
        json.dump(duplicates, file)
        file.close()

class DuplicateViewer(object):
    def __init__(self, duplicate_file):
        with open(duplicate_file) as file:
            print("[Reading duplicate file...]")
            duplicates = json.load(file)
            print("[Duplicate file read into dictionary]")
            self.window = "Duplicate Viewer"
            self.display(duplicates)

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
    def get_file_size_text(path: str, unit = SizeUnit.MB):
        size = os.path.getsize(path)
        converted_size, suffix = DuplicateViewer.convert_unit(size, unit)
        return str(round(converted_size, 2)) + " " + suffix

    def display(self, duplicates):
        cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        x, y, screen_width, screen_height = cv2.getWindowImageRect(self.window)
        cv2.destroyAllWindows()
        print(screen_width, screen_height)
        image_width = int(screen_width / 2)
        self.running = True
        for duplicate in duplicates:
            if self.running:
                similarity = duplicate["similarity"]
                paths = duplicate["paths"]
                picture1 = Picture(paths[0], True, image_width)
                picture2 = Picture(paths[1], False, image_width)
                if (not picture1.image is None) and (not picture2.image is None):
                    picture1.setup(screen_height, similarity)
                    picture2.setup(screen_height, similarity)
                    pictures = [picture1, picture2]
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
                        elif k == ord('1'):
                            pictures[0].border = not pictures[0].border
                            self.update_display(pictures)
                        elif k == ord('2'):
                            pictures[1].border = not pictures[1].border
                            self.update_display(pictures)
                        elif k == ord(' '):
                            for picture in pictures:
                                if not picture.border:
                                    if Path(picture.path).is_file():
                                        try:
                                            send2trash(picture.path)
                                            print("[Deleting]: " + picture.path)
                                        except:
                                            print("[Failed to delete]: " + picture.path)
                            print("----------------------------")
                            break
                else:
                    # Notify the user that an image cannot be displayed
                    print("[Skipping]: (" + picture1.path + ", " + picture2.path + ")")

    def update_display(self, pictures):
        for picture in pictures:
            if picture.border:
                picture.add_border()
            else:
                picture.reset()
        cv2.imshow(self.window, cv2.hconcat([pictures[0].image, pictures[1].image]))

duplicate_file = 'all_duplicates.json'

finder = DuplicateFinder(["/mnt/d/Media/All"],
    #"/mnt/c/Dev/duplicate_finder/data"],
    #"/mnt/c/Dev/duplicate_finder/data_original"],
    duplicate_file,
    10)
#viewer = DuplicateViewer(duplicate_file)