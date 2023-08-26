import os
import cv2
import numpy as np
import json
from PIL import Image
import dhash
import pyheif
from pathlib import Path

class DuplicateFinder(object):
    def __init__(self, directories, save_file):
        self.save_file = save_file
        self.files = []
        self.duplicates = {}
        self.directories = directories

    def start(self):
        self.retrieve_files()
        self.retrieve_duplicates()

    def retrieve_files(self):
        for directory in self.directories:
            directory_path = Path(directory)
            if directory_path.is_dir():
                print("Processing directory: " + str(directory))
                for parent_directory, _, files in os.walk(directory):
                    for file in files:
                        path = os.path.join(parent_directory, file)
                        if os.path.exists(path):
                            self.files.append(path)
            else:
                print("Could not process directory: " + str(directory))
        print("Found " + str(len(self.files)) + " files in all directories")

    def retrieve_duplicates(self):
        file_count = len(self.files)
        for index, path in enumerate(self.files):
            print("Processing " + str(path) + " (" + str(index) + "/" + str(file_count) + ")")
            try:
                try:
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                except:
                    try:
                        capture = cv2.VideoCapture(path)
                        success, image = capture.read()
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        capture.release()
                    except:
                        try:
                            heif_file = pyheif.read(path)
                            image = Image.frombytes(
                                heif_file.mode, 
                                heif_file.size, 
                                heif_file.data,
                                "raw",
                                heif_file.mode,
                                heif_file.stride,
                            )
                            image = np.array(image)
                        except:
                            print("Could not recognize file format for: " + path)
                try:
                    _ = image.shape
                    resized = cv2.resize(image, (8, 8))
                    #file_hash = str(cv2.img_hash.AverageHash_create().compute(resized).tobytes())
                    array = Image.fromarray(resized)
                    file_hash = dhash.dhash_int(array, size=8)
                    if file_hash in self.duplicates:
                        self.duplicates[file_hash].append(path)
                    else:
                        self.duplicates[file_hash] = [path]
                except:
                    pass
            except KeyboardInterrupt:
                self.save_to_file()
            
    def save_to_file(self):
        f = open(self.save_file, "w")
        json.dump(self.duplicates, f)
        f.close()

finder = DuplicateFinder(
    [r'/mnt/d/Media/Nicole/', r'/mnt/d/Media/All'], 
    "heic_duplicates.json"
)

finder.start()

finder.save_to_file()

print("Exiting program")