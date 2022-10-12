from utils.preprocessing import spacy_preprocessor
from utils.vectorization_utilis import generate_X_train_Y_train

import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader


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

if __name__ == '__main__':
    preprocessor = spacy_preprocessor()
    X_train, Y_train = generate_X_train_Y_train(preprocessor,"/home/euclide03/my_chatbot_project/intents.json")
    dataset = Chatbot_Dataset(X_train,Y_train)
    train_loader=DataLoader(dataset=dataset,batch_size=8, shuffle=True, num_workers=2)




