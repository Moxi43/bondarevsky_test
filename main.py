from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from net import Net

#samples per batch
batch_size = 10
#training set and validation
test_size = 0.3
valid_size = 0.1

#declaration of transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

data = datasets.ImageFolder("data", transform=transform)

#For test
num_data = len(data)
indices_data = list(range(num_data))
np.random.shuffle(indices_data)
split_tt = int(np.floor(test_size * num_data))
train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

#For Valid
num_train = len(train_idx)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_data))
train_new_idx, valid_idx = indices_train[split_tt:], indices_train[:split_tt]

#samplers for train and valid batches
train_sampler = SubsetRandomSampler(train_new_idx)
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


#define loaders
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, \
    sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, \
    sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(data, sampler=test_sampler,   \
    batch_size=batch_size, num_workers=1)

#define classes
classes = [0, 1]


#CNN model
model=Net()

#move tensors to gpu
if torch.cuda.is_available():
    model.cuda()


#loss funcion
loss_func = torch.nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


#TRAINING
EPOCHS = 7

valid_loss_min = np.Inf

for epoch in range(1, EPOCHS+1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        #move to gpu
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        #clear the gradients
        optimizer.zero_grad()

        #compute predicted outputs
        output = model(data)

        #batch loss
        loss = loss_func(output, target)

        #compute gradient of the loss
        loss.backward()

        #optimization step
        optimizer.step()

        #update training loss
        train_loss += loss.item()*data.size(0)

#VALIDATING
    model.eval()
    for data, target in valid_loader:
        #move to gpu
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        #compute predicted outputs
        output = model(data)
        #calc the batch loss
        loss = loss_func(output, target)
        #update average val loss
        valid_loss += loss.item()*data.size(0)

    #average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_los = valid_loss/len(valid_loader.dataset)

    #train/val stats
    print("Epoch: {} \tTraining Loss: {:.2f} \tValidation Loss: {:.2f}" \
        .format(epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print("Validation loss decreased {:.2f} --> {:.2f}. Saving..".format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), "model.pt")
        valid_loss_min = valid_loss


#TEST
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()
i=1
# iterate over test data
len(test_loader)
for data, target in test_loader:
    i=i+1
    if len(target)!=batch_size:
        continue

    # move tensors to GPU if CUDA is available
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = loss_func(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    
    print("#########################")
    print(pred)
    print(correct_tensor)
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class

    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
