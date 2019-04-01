# Facial Keypoint Detection

This project is about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces.

This project is directly inspired by the @Udacity Computer Vision Nanodegree and has been modified in my way in <a href="https://pytorch.org/get-started/locally/">Pytorch</a>.

## File Description
- custom_dataset.py: Create the dataset using the Dataset class from Pytorch
- inference.ipynb: Notebook for inference
- network.py: Create the architecture of the CNN according to <a href="https://arxiv.org/pdf/1710.00977.pdf">this paper</a>
- training.ipynb: Notebook using Google Colab GPU to train the CNN
- transformation.py: Python file that contains all classes to transform the data (images and keypoints)

## Authors
Medhy Vinceslas

## License
LICENSE: This project is licensed under the terms of the MIT license.

## Acknowledgement
Thank you to the @udacity staff for giving me the opportunity to improve my skills in Computer Vision.