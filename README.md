# Facial Keypoint Detection

<img src="keypts.png">

This project is about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces.

This project is directly inspired by the @Udacity Computer Vision Nanodegree and has been modified in my way in <a href="https://pytorch.org/get-started/locally/">Pytorch</a>.

## File Description
- custom_dataset.py: Create the dataset using the Dataset class from Pytorch
- inference.ipynb: Notebook for inference
- network.py: Create the architecture of the CNN according to <a href="https://arxiv.org/pdf/1710.00977.pdf">this paper</a>
- training.ipynb: Notebook using Google Colab GPU to train the CNN
- transformation.py: Python file that contains all classes to transform the data (images and keypoints)

## Pipeline

### Customized Dataset
```
class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
```

### Preprocessing/Transform (Images and Keypoints)
```
class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        
    pass



class Rescale(object):
    """Rescale the image in a sample to a given size.  

    Args:  
        output_size (tuple or int): Desired output size. If tuple, output is  
        matched to output_size. If int, smaller of image edges is matched  
        to output_size keeping aspect ratio the same.  
    """
    pass  



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    pass



class ToTensor(object):
    pass



transform = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    Normalize(),
    ToTensor()
])
```

### Prepare Validation and Load Data
```
def train_valid_split(training_set, validation_size):
    """ Function that split our dataset into train and validation
        given in parameter the training set and the % of sample for validation"""
    
    pass




train_set = FacialKeypointsDataset(csv_file=csv_file,
                                   root_dir=root_dir,
                                   transform=transform)

train_sampler, valid_sampler = train_valid_split(train_set, valid_size)


train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=num_workers)
```

### Build the CNN Architecture
```
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__() 
        pass
        
        

        
    def forward(self, x):
        pass
```

### Train

### Select Region of Interest

### Inference


## Authors
Medhy Vinceslas

## License
LICENSE: This project is licensed under the terms of the MIT license.

## Acknowledgement
Thank you to the @udacity staff for giving me the opportunity to improve my skills in Computer Vision.