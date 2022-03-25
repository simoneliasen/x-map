import torch

#net kræver self.testloader, self.model
def test(net):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in net.testloader:
                
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.to('cuda'), labels.to('cuda')
                    
                outputs = net.model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: ', 100*(correct/total), '%')
        print('Correct: ', correct, ' Total: ', total)