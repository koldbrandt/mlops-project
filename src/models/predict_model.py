import torch
from torch import nn
import model

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
        for images, labels in testloader:

            images = images.resize_(images.size()[0], 784)

            output = model.forward(images)

            labels = labels.type(torch.LongTensor)

            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = labels.data == ps.max(1)[1]
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
        saved_model = model.MyAwesomeModel(
            checkpoint["input_size"],
            checkpoint["output_size"],
            checkpoint["hidden_layers"],
        )
        saved_model.load_state_dict(checkpoint["state_dict"])

        return saved_model

if __name__ == "__main__":
    Evaluate()