# Object Detection using Low-light Images
This project utilizes the Tensorflow Object Detection API to identify objects in images taken from the ExDark dataset.

## Structuring the data
The original structure of the dataset is modified to look like this:

```
/low-light-detection
├── data
│   ├── annotations
│   ├── test
│   ├── train
```

where `annotations` contains all the annotation text files, `test` contains ~20% of the images from each of the 12 classes, and `train` contains ~80% of the images from each class.

## Converting annotations to CSV
The annotations are converted to CSV using `txt_to_csv.py` so that they can be used to generate the required TFRecord files. From within the repo folder (/low-light-detection), run:

`python txt_to_csv.py`

This will generate `train_labels.csv`. Run the same script after modifying all strings containing "train" to "test" to generate `test_labels.csv`. Then, move both CSV files to `data`.

## Generating TFRecord files
From within the repo folder, run:

`python generate_tfrecord.py --csv_input=data/train_labels.csv --image_dir=data/train --output_path=train.record`

`python generate_tfrecord.py --csv_input=data/test_labels.csv --image_dir=data/test --output_path=test.record`

This will generate the record files needed for training the model.

## Running the Jupyter notebook
The record files and `labelmap.pbtxt` need to be made available for training before executing the notebook. The method here involves uploading them to Google Drive and loading them from there.

Specific instructions and explanations about the code can be found in the notebook.

## Dependencies and configurations
### Local
- Python: 3.5.4 (64-bit)
- numpy: 1.17.1
- Pillow: 6.1.0
- pandas: 0.25.1
- tensorflow: 1.14.0
- object-detection: 0.1

### Google Colab
- Runtime type: Python 3
- Hardware accelerator: GPU

## References
1. [Exclusively-Dark-Image-Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
2. [Creating your own object detector](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85)
3. [How to train your own Object Detector with TensorFlow’s Object Detector API](http://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
4. [Object Detection in Google Colab with Custom Dataset](https://hackernoon.com/object-detection-in-google-colab-with-custom-dataset-5a7bb2b0e97e)