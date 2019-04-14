import torch
import torch.nn as nn
import torch.nn.functional as F

# can use the below import to choose to initialize the weights of the Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2) #(window size, stride)
                
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4) #224, 32
        
        self.conv2 = nn.Conv2d(32, 64, 3) #112, 64
        
        self.conv3 = nn.Conv2d(64, 128, 2) #56, 128
        
        self.conv4 = nn.Conv2d(128, 256, 1) #28, 256
        
        self.fc1 = nn.Linear(43264, 1000)   
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
        

        
    def forward(self, x):
      
      
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x

    
    
    
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2) #(window size, stride)
                
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        self.conv3 = nn.Conv2d(64, 128, 4)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.fc1 = nn.Linear(30976, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
        

        
    def forward(self, x):
      
      
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x