import matplotlib.pyplot as plt
import numpy as np

def visualize(num_images, train_loader, row, col):
  # obtain one batch of training images
  batch = next(iter(train_loader))

  # display n images
  for i in np.arange(num_images):

      images, labels = batch['image'], batch['keypoints']

      #unormalize images
      image = images[i].numpy()
      image = np.transpose(image, (1, 2, 0))

      #unormalize labels
      labels = labels[i].numpy()
      labels = labels*50.0+100

      plt.subplot(row,col,i+1)
      plt.imshow(np.squeeze(image), cmap='gray')
      plt.scatter(labels[:, 0], labels[:, 1], s=20, marker='.', c='m')
      plt.axis('off')