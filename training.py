def train(n_epochs, save_location_path):
    print('Weight initialization ...')
    train_on_gpu = torch.cuda.is_available()

    #def init_weights(m):
        #if type(m) == nn.Linear:
            #I.xavier_uniform(m.weight)
    
    # takes in a module and applies the specified weight initialization
    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
    

    model = Net()
    model.apply(weights_init_uniform_rule)
    #model.apply(init_weights)

    if train_on_gpu:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 1e-5)


    valid_loss_min = np.Inf
    
    train_loss_record = []
    val_loss_record = []
    
    print('Start training')
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
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)


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
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            output = model(images)
            loss = criterion(output, key_pts)
          
            valid_loss += loss.item()*images.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(valid_loader)

        # print training/validation statistics 
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...")
            torch.save(model.state_dict(), save_location_path)
            valid_loss_min = valid_loss
            
       
        train_loss_record.append(train_loss)
        val_loss_record.append(valid_loss)
        
    return train_loss_record, val_loss_record