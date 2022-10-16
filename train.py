from utils.preprocessing import spacy_preprocessor
from utils.vectorization_utilis import generate_X_train_Y_train
import model
from model import NeuralNetwork

import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

import argparse
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Chatbot_Dataset(Dataset):
    """Chatbot dataset."""

    def __init__(self, X_train,y_train):
        """
        Args:
            y_train: np_array of labels.
            x_train: (string): np_array of samples.
        """
        self.x_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_train[idx], self.y_train[idx]

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Build data loader
    preprocessor = spacy_preprocessor()
    X_train, Y_train, bag_of_words = generate_X_train_Y_train(preprocessor, args.json_path)
    dataset = Chatbot_Dataset(X_train, Y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    input_size = len(X_train[0])
    output_size = args.num_classes

    # Build the models
    model = NeuralNetwork(input_size, args.hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the models
    for epoch in range(args.num_epochs):
        for (inputs, labels) in train_loader :

            # Set mini-batch dataset
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            # Forward, backward and optimize
            predictions = model(inputs.float())
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (epoch+1) % args.log_step == 0:
                print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
    model_data={
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_classes": args.num_classes,
        "bag_of_words": bag_of_words
    }
    torch.save(model_data, os.path.join(
        args.model_path, 'model_data-{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('json_path', type=str)
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=8, help='dimension of layer hidden states')
    parser.add_argument('--num_classes', type=int, default=9, help='number of tags')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)