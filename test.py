from helper_methods import imshow
import torchvision
import torch
from model import Net


def testing(testloader, classes, PATH):

    dataiter = iter(testloader) # probably expendable, once we are sure that imshow works
    images, labels = dataiter.next() #  probably expendable, once we are sure that imshow works

    #print some images images
    imshow(torchvision.utils.make_grid(images))

    #load back in our saved model 
    net = Net()
    net.load_state_dict(torch.load(PATH))

    #Variable for total model performance
    correct = 0
    total = 0

    #What are the classes that performed well and not?
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print('Finished categorical testing')
