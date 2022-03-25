import torch

#net kræver self.trainloader, self. optimizer, self.model, self.criterion, self.scheduler
def train(net):
        EPOCHS = 2 #fra 200
        for epoch in range(EPOCHS):
            losses = []
            running_loss = 0
            for i, inp in enumerate(net.trainloader):
                
                inputs, labels = inp
                if torch.cuda.is_available():
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    
                net.optimizer.zero_grad()
            
                outputs = net.model(inputs)
                loss = net.criterion(outputs, labels)
                losses.append(loss.item())

                loss.backward()
                net.optimizer.step()
                
                running_loss += loss.item()
                
                if i%1 == 0 and i > 0:
                    print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 1)
                    running_loss = 0.0

            avg_loss = sum(losses)/len(losses)
            net.scheduler.step(avg_loss)
                    
        print('Training Done')