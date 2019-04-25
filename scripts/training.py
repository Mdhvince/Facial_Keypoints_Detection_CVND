#!/usr/bin/python3

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models
import torch.optim as optim
import torch.nn.init as I

from load_data import *
from network import Net

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from functools import partial
from threading import Thread
from tornado import gen


batch_size = 128
num_workers = 4
valid_size = 0.2
csv_file = '../data/training_frames_keypoints.csv'
root_dir = '../data/training/'
save_location_path = 'saved_models/modelx.pt'
n_epochs = 600

train_set = create_dataset(csv_file, root_dir)
train_sampler, valid_sampler = train_valid_split(train_set, valid_size)
train_loader, valid_loader = build_lodaers(train_set, train_sampler, valid_sampler,
                                           batch_size, valid_size,
                                           num_workers, csv_file, root_dir)
#visualize(20, train_loader, 4, 5)


source = ColumnDataSource(data={'epochs': [],
                                'trainlosses': [],
                                'vallosses': [] }
)

plot = figure(plot_width=1000)
plot.line(x= 'epochs', y='trainlosses',
          color='green', alpha=0.8, legend='Train loss', line_width=2,
          source=source)

plot.line(x= 'epochs', y='vallosses',
          color='red', alpha=0.8, legend='Val loss', line_width=2,
          source=source)

#train(n_epochs, train_loader, valid_loader, save_location_path)

# - Add plot to the current doc
doc = curdoc()
doc.add_root(plot)

@gen.coroutine
def update(new_data):
    source.stream(new_data)

def train(n_epochs=n_epochs,
          train_loader=train_loader, valid_loader=valid_loader,
          save_location_path=save_location_path):

    train_on_gpu = torch.cuda.is_available() 
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            I.xavier_uniform_(m.weight)
    

    model = Net()
    model.apply(init_weights)

    if train_on_gpu:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

    valid_loss_min = np.Inf
    
    model.train()

    for epoch in range(1, n_epochs+1):
        # Keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        for data in train_loader:

            # Grab the image and its corresponding label
            images = data['image']
            key_pts = data['keypoints']

            if train_on_gpu:
                images, key_pts = images.cuda(), key_pts.cuda()


            # Flatten keypoints & convert data to FloatTensor for regression loss
            key_pts = key_pts.view(key_pts.size(0), -1)
            if train_on_gpu:
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)
            else:
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)


            optimizer.zero_grad()                           # Clear the gradient        
            output = model(images)                          # Forward
            loss = criterion(output, key_pts)               # Compute the loss
            loss.backward()                                 # Compute the gradient
            optimizer.step()                                # Perform updates using calculated gradients

            train_loss += loss.item()*images.size(0)
        
        # Validation
        model.eval()
        for data in valid_loader:

            images = data['image']
            key_pts = data['keypoints']

            if train_on_gpu:
                images, key_pts = images.cuda(), key_pts.cuda()


            key_pts = key_pts.view(key_pts.size(0), -1)
            if train_on_gpu:
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)
            else:
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

            output = model(images)
            loss = criterion(output, key_pts)
          
            valid_loss += loss.item()*images.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(valid_loader)

        new_data = {'epochs': [epoch],
                    'trainlosses': [train_loss],
                    'vallosses': [valid_loss] }

        doc.add_next_tick_callback(partial(update, new_data))

        for key, val in new_data.items():
            print(f"{key}: {val}")
            print(" ")

#global update
# - Call the train() function each 1s (1000ms)
#doc.add_periodic_callback(train, 1000)
thread = Thread(target=train)
thread.start()



