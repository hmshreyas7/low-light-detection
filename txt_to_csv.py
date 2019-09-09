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

root = "./data"
type = "train"
arguments = sys.argv
length = len(arguments)

if(length == 2):
    type = arguments[1]
elif(length > 2):
    raise Exception("Too many arguments")

imgs = os.listdir(root + "/" + type)
with open(type + "_labels.csv", "w") as f:
    f.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
f.close()

for img in imgs:
    image = Image.open(root + "/" + type + "/" + img)
    height, width = np.array(image).shape[:2]
    annotations_file = os.path.join(root + "/annotations/" + img + ".txt")
    annotations = ""

    with open(annotations_file, "r") as f:
        annotations = f.read()
    f.close()

    annotations = annotations[17:]
    objects = annotations.split("\n")
    objects = objects[:-1]

    for object in objects:
        values = object.split(" ")
        label, xmin, ymin = values[:3]
        xmax = str(int(xmin) + int(values[3]))
        ymax = str(int(ymin) + int(values[4]))

        with open(type + "_labels.csv", "a") as f:
            f.write(img + "," + str(width) + "," + str(height) + "," + label + "," + xmin + "," + ymin + "," + xmax+ "," + ymax + "\n")
        f.close()