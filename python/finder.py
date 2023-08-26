from pathlib import Path
import cv2
import json
import sys
from PIL import Image, ImageOps
import numpy as np
import dhash
try:
    import pyheif # only on UNIX
except ImportError:
    pass

class DuplicatePair(dict):
    def __init__(self, hash1, path1: Path, hash2, path2: Path, similarity: int):
        self.hash1 = hash1
        self.path1 = str(path1)
        self.hash2 = hash2
        self.path2 = str(path2)
        self.similarity = similarity
        dict.__init__(self, paths=[self.path1, self.path2], similarity=self.similarity)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.hash1 == other.hash1 and self.hash2 == other.hash2) or \
                   (self.hash1 == other.hash2 and self.hash2 == other.hash1)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class DuplicateFinder(object):
    def __init__(self, file_directories, dictionary_destination_file: str, similarity_threshold: int):
        self.directories = file_directories
        self.save_file = dictionary_destination_file
        self.similarity_threshold = similarity_threshold # %
        self.find_paths()
        self.find_hashes()
        self.find_duplicates()
        self.save_to_file()
    
    def find_paths(self):
        self.paths = []
        for directory in self.directories:
            directory = Path(directory)
            print("Processing directory: " + str(directory))
            if directory.is_dir():
                # Recurse through all subdirectories and return all filepaths inside of them.
                for path in directory.rglob("*"):
                    assert path.is_file()
                    self.paths.append(path)
                print("Processed directory.")
            else:
                print(str(directory) + " is not a valid directory")

    def find_hashes(self):
        print("Finding hashes...")
        self.hashes = []
        path_count = len(self.paths) / 100
        for index, path in enumerate(self.paths):
            image = self.find_image(str(path))
            if image is not None:
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grey, (8, 8))
                array = Image.fromarray(resized)
                hash = dhash.dhash_int(array, size=8)
                self.hashes.append((hash, path))
            if index % 10 == 0:
                print("~" + str(round(index / path_count)) + "% of hashes found")

    def find_image(self, file: str):
        image = None
        path = Path(file)
        if path.is_file():
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

    def find_duplicates(self):
        self.duplicates = []
        hash_count = len(self.hashes) / 100
        counter = 0
        for hash1, path1 in self.hashes:
            for hash2, path2 in self.hashes:
                if path1 != path2:
                    hamming_distance = dhash.get_num_bits_different(hash1, hash2)
                    if hamming_distance <= self.similarity_threshold:
                        pair = DuplicatePair(hash1, path1, hash2, path2, int(100 - hamming_distance / (self.similarity_threshold + 1) * 100))
                        if not self.identical_pair_exists(pair):
                            self.duplicates.append(pair)
            counter += 1
            if counter % 10 == 0:
                print("~" + str(round(counter / hash_count)) + "% of duplicates processed")
            # file = str(path1)
            # if hash1 in self.duplicates:
            #     self.duplicates[hash1].append(file)
            # else:
            #     self.duplicates[hash1] = [file]

    def identical_pair_exists(self, pair):
        for duplicate_pair in self.duplicates:
            if duplicate_pair == pair:
                return True
        return False

    def save_to_file(self):
        file = open(self.save_file, "w")
        json.dump(self.duplicates, file)
        file.close()

finder = DuplicateFinder([
    #"/mnt/c/Dev/duplicate_finder/data"],
    "c:/Dev/duplicate_finder/data"],
    'heic_duplicates.json',
    10)