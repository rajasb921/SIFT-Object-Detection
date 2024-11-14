<a name="readme-top"></a>
<br />
<h1 align="center">Rocky Detection System</h1>

  <p align="center">
    A computer vision system for detecting and analyzing object transformations using SIFT features
  </p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#technical-details">Technical Details</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This project implements an object detection system that can identify a specific object (Rocky) in different images and determine its position, orientation, and scale. The system uses Scale-Invariant Feature Transform (SIFT) for feature detection and matching, making it robust to various transformations including rotation, scaling, and translation.

Key features include:
- SIFT-based feature detection and matching
- Scale and rotation invariant object detection
- Precise center point calculation
- Object height estimation
- Orientation measurement
- Memory-efficient batch processing for large images

### Built With
* Python
* OpenCV
* NumPy

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
- Python 3.x
- pip package manager

### Installation
1. Clone the repository
2. Install required packages:
```sh
pip install opencv-python numpy
```
3. Ensure you have a reference image named 'reference.png' in the project directory

<!-- USAGE -->
## Usage

Run the detection script:
```sh
python project2.py
```

The program will:
1. Prompt for an input image filename
2. Process the image to detect the target object
3. Output results in the format:
```
<center_x> <center_y> <height> <angle>
```
Where:
- center_x, center_y: Coordinates of the object's center
- height: Estimated height of the object
- angle: Orientation angle in degrees

If the object is not found, the program outputs "0 0 0 0"

<!-- TECHNICAL DETAILS -->
## Technical Details

### Feature Detection and Matching
- Uses SIFT (Scale-Invariant Feature Transform) algorithm
- Configurable for up to 10,000 features per image
- Implements nearest neighbor ratio test for match filtering
- Uses a ratio threshold of 0.4 for optimal matching

### Image Processing Pipeline
1. **Image Loading and Preprocessing**
   - Loads reference and test images in grayscale
   - Resizes reference image to match test image height
   - Maintains aspect ratio during resize

2. **Feature Extraction**
   - Computes SIFT keypoints and descriptors
   - Processes descriptors in batches of 100 for memory efficiency

3. **Match Filtering**
   - Applies ratio test to filter high-quality matches
   - Requires minimum 2 good matches for detection
   - Uses top 10 matches for transformation analysis

4. **Transformation Analysis**
   - Calculates object center using averaged keypoint coordinates
   - Determines scale by comparing keypoint distances
   - Computes orientation through angle differences between matched points
   - Uses Euclidean distance for scale estimation

### Key Functions
- `ratio_test_match`: Implements nearest neighbor ratio test
- `calculate_angle`: Determines orientation angles (0-360 degrees)
- `euclidean_distance`: Calculates distances for scale estimation

The system assumes a known reference height of 1600 pixels and scales the output height accordingly based on detected transformations.
