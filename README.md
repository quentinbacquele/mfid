# MFID (MonkeyFaceID)

## Introduction

The MFID (MonkeyFaceID) System is a python package designed to recognize Rhesus macaques in videos and images using computer vision. It comprises three main components: body detection, face detection, and identity detection.

## Table of Contents

- Introduction
- Installation
- Usage
- Features
- Requirements
- Dependencies


## Installation

To install MFID, ensure you have Python 3.6 or later installed on your system. Follow these steps:

1. Clone the MFID repository:

```bash
git clone https://github.com/quentinbacquele/mfid.git
```

2. Navigate to the cloned directory:

```bash
cd mfid
```

3. Install the required dependencies:

```bash
pip install -e .
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Optionnal step for GPU usage (if not already done)

Install PyTorch with the correct CUDA, cuDNN, and GPU support. Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).


## Usage

To launch the MFID application, execute the following command from the terminal:

```bash
mfid
```

This command activates the App Launcher, providing access to various functionalities, including body detection, face detection, and annotation tools.

## Features

**Body Detection**: Sorts videos and images based on monkey presence.
**Face Detection and Annotation**: Detects monkeys face and generates datasets with cropped faces and provides tools for both manual and automatic annotation of cropped images.
**Identity Recognition**: Recognizes and annotate monkeys.

<details>
  <summary><h2>Body App Instructions</h2></summary>

- **Model**: Choose the model for detection. The smaller the model, the faster the inference. The larger the model, the better the accuracy.
- **Confidence Threshold**: Set the confidence threshold for detection.
- **IOU Threshold**: Set the intersection-over-union (IOU) threshold for non-maximum suppression (NMS).
- **Show**: Check this box to show the video with detections in real-time.
- **Save**: Check this box to save the videos with monkey detections.
- **Save TXT**: Check this box to save the detection results, including bounding box coordinates, in a text file.
</details>

<details>
  <summary><h2>Face App Instructions</h2></summary>

- **Select save folder**: Choose the folder to save the datasets produced.

### Detection

- **Load videos folder**: Load the folder with images or videos.
- **Save coordinates**: Check this box to save the coordinates of the bounding boxes of the detected faces.
- **Save full frames**: Uncheck it if you do not use the manual annotation.
- **Skip frames**: Choose the interval for detection.
- **Run detection**: Run the model for face detection.


### Annotation

- **Load folder with extracted faces**: Allows to continue annotation of cropped faces when the detection was already done before.
- **Annotate cropped faces**: Launchs the manual annotator.
- **Automatic annotate**: Sorts cropped faces in the selected folder by keyword.
- **Delete full frames**: Deletes full frames in the folder once the manual annotation is done.
</details>

<details>
  <summary><h2>Identity App Instructions</h2></summary>

## Identity App Instructions

- **Load video/image**: Choose the file for identity detection.
- **Run detection**: Run the model and provide the image or the video annotated as an output.
</details>

## Requirements

- Git
- Python 3.x
- CUDA, cuDNN, and compatible GPU (if you plan to use GPU for inference)

<details>
  <summary><h2>Dependencies</h2></summary>

MFID requires the following libraries:

- PyQt5
- ultralytics
- numpy
- opencv-python
- Pillow
- PyYAML
- requests
- scipy
- torch
- torchvision
- tqdm
- pandas
- seaborn
</details>


