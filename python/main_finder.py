import os
import cv2
import numpy as np
import json
from PIL import Image
import pyheif


class DuplicateFinder(object):
    def __init__(self, directory):
        self.duplicates = {}
        self.directory = directory

    def start(self):
        self.duplicates = self.retrieve_duplicates()

    def retrieve_duplicates(self):
        duplicates = {}
        if os.path.exists(self.directory):
            for dir, _, files in os.walk(self.directory):
                file_count = len(files)
                print('Scanning %s...' % dir)
                for number, file in enumerate(files):
                    path = os.path.join(dir, file)
                    if not os.path.exists(path):
                        continue
                    print(str(round(number / file_count * 100)) + "%" + " complete")
                    try:
                        image = cv2.imread(path)
                    except:
                        try:
                            capture = cv2.VideoCapture(path)
                            success, image = capture.read()
                            capture.release()
                            assert success, "Failed to read from video capture"
                        except:
                            try:
                                heif_file = pyheif.read(path)
                                image_file = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride,)
                                image = np.array(image_file)
                            except:
                                print("Failed to process file " + path)
                    try:
                        _ = image.shape # test for exceptions
                        file_hash = str(cv2.img_hash.BlockMeanHash_create().compute(image).tobytes())
                        if file_hash in duplicates:
                            duplicates[file_hash].append(path)
                        else:
                            duplicates[file_hash] = [path]
                    except:
                        pass
        else:
            print('%s is not a valid directory, please verify' % directory)
        return duplicates

    def save_to_file(self):
        f = open("C:/Users/Martin/Desktop/duplicate_finder/data_test.json", "w")
        json.dump(self.duplicates, f)
        f.close()

finder = DuplicateFinder('C:/Users/Martin/Desktop/duplicate_finder/data')

finder.start()

finder.save_to_file()

print("Exiting program")