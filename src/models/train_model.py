import pathlib
from model import Network
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import hydra
import os
import wandb
import argparse



#@hydra.main(config_name= "training_conf.yaml" ,config_path="../../conf")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    print("Working directory : {}".format(os.getcwd()))
    print("Training day and night")    
    model = Network()   


    # Magic
    wandb.watch(model, log_freq=cfg.print_every)

    model.train()

    trainloader = torch.load(cfg.train_data)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()

    steps = 0
    running_loss = 0
    losses = []
    timestamp = []
    epochs = cfg.epochs
    print_every = cfg.print_every
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            #images.resize_(images.size()[0], 784)
            optimizer.zero_grad()

            labels = labels.type(torch.LongTensor)
            

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                wandb.log({"loss": loss})

                # Model in inference mode, dropout is off
                model.eval()
                
                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every)
                )

                losses.append(running_loss / print_every)
                timestamp.append(steps)
                running_loss = 0
                # Make sure dropout and grads are on for training
                model.train()
    plt.plot(timestamp, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training.png")

    #plt.show()
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "models/checkpoint.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    wandb.init(config= args)
    main(wandb.Config)
