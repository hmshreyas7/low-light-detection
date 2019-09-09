"""
Usage:
  # Create train labels:
  python txt_to_csv.py train
  
  # Create test labels:
  python txt_to_csv.py test
"""
import os
import numpy as np
from PIL import Image
import sys

# Set root directory and default labels type
root = "./data"
type = "train"

# Get list of command line arguments and its length
arguments = sys.argv
length = len(arguments)

# Handle different argument lengths
if(length == 2):
    type = arguments[1]
elif(length > 2):
    raise Exception("Too many arguments")

# Get image directory based on type and open CSV file for writing
imgs = os.listdir(root + "/" + type)
with open(type + "_labels.csv", "w") as f:
    f.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
f.close()

# For each image in directory, append required values to CSV file
for img in imgs:
    image = Image.open(root + "/" + type + "/" + img)
    height, width = np.array(image).shape[:2]
    annotations_file = os.path.join(root + "/annotations/" + img + ".txt")
    annotations = ""

    with open(annotations_file, "r") as f:
        annotations = f.read()
    f.close()

    # Ignore first 16 characters (Annotation tool information) and get each object's data
    annotations = annotations[17:]
    objects = annotations.split("\n")
    objects = objects[:-1]

    # For each object in the image, get class label and coordinates
    for object in objects:
        values = object.split(" ")
        label, xmin, ymin = values[:3]
        xmax = str(int(xmin) + int(values[3]))
        ymax = str(int(ymin) + int(values[4]))

        with open(type + "_labels.csv", "a") as f:
            f.write(img + "," + str(width) + "," + str(height) + "," + label + "," + xmin + "," + ymin + "," + xmax+ "," + ymax + "\n")
        f.close()