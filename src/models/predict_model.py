import torch
from torch import nn
from model import Network
import wandb
wandb.init()

class Evaluate(object):
    def __init__(self):
        """
        Validates the given model with the given test set.
        """
        
        model = self.load_checkpoint("checkpoint.pth")
        testloader = torch.load("data/processed/test.pt")
        criterion = nn.NLLLoss()
        accuracy = 0
        test_loss = 0
        first = True
        for images, labels in testloader:
            output = model.forward(images)

            labels = labels.type(torch.LongTensor)

            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = labels.data == ps.max(1)[1]

            if first:
                my_table = wandb.Table()
                my_table.add_column("image", [images.unsqueeze(1)[0]])
                my_table.add_column("label", labels.data[0])
                my_table.add_column("class_prediction", ps.max(1)[1])

                # Log your Table to W&B
                wandb.log({"mnist_predictions": my_table})
                first = False
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(
            "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
        )

    def load_checkpoint(self, modelname):     
        """
        Loads saved model and returns it
        """   
        checkpoint = torch.load("models/" + modelname)
        saved_model = Network()
        saved_model.load_state_dict(checkpoint["state_dict"])

        return saved_model

if __name__ == "__main__":
    Evaluate()