## Counterfeit Currency Detector for Indian 500 and 2000 Rupees Notes

This project aims to detect counterfeit Indian 500 and 2000 rupees notes using Python. The project leverages ORB (Oriented FAST and Rotated BRIEF) and SSIM (Structural Similarity Index) for image processing and comparison.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Counterfeit currency detection is a significant problem for the economy. This project uses image processing techniques to identify fake notes. The system compares features of the input currency note with a reference genuine note to determine its authenticity.

## Features

- **Detection of 500 and 2000 rupees notes**: Specialized for Indian currency.
- **ORB (Oriented FAST and Rotated BRIEF)**: Used for feature detection.
- **SSIM (Structural Similarity Index)**: Used for comparing the structural similarity of images.
- **User-friendly GUI**: Simple graphical user interface for ease of use.

## Technologies Used

- **Python**: Core programming language.
- **OpenCV**: Library for image processing.
- **scikit-image**: Library for SSIM computation.
- **Tkinter**: Used for GUI development.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KumarSatyam24/counterfeit-currency-detector.git
    ```

2. Navigate to the project directory:

    ```bash
    cd counterfeit-currency-detector
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main application script:

    ```bash
    python controller.ipynb
    ```

2. Follow the GUI prompts to upload an image of the currency note you wish to verify.

## Project Structure

- **controller.ipynb**: Main script for controlling the detection logic.
- **500_Testing.ipynb**: Script for testing 500 rupees notes.
- **2000_Testing.ipynb**: Script for testing 500 rupees notes.
- **gui_2.ipynb**: Script for the GUI interface.
- **edge_detection.py**: script to crop the image of the currency note taken by the camera.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

