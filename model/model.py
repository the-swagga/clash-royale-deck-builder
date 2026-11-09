import json
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class DeckBuilderDataset(Dataset):
    def __init__(self, path, num_cards, is_train, min_masked=1, max_masked=7, seed=38):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        decks = []
        weights = []
        for record in records:
            decks.append(record["deck"])
            weights.append(record["weight"])

        self.decks = decks
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_cards = num_cards
        self.is_train = is_train
        self.min_masked = min_masked
        self.max_masked = max_masked
        self.seed = seed

    def __len__(self):
        return len(self.decks)

    def __getitem__(self, index):
        if self.is_train:
            rng = random.Random()
        else:
            rng = random.Random(self.seed + index)

        deck = self.decks[index]
        num_cards_masked = rng.randint(self.min_masked, self.max_masked)
        masked_cards_positions = set(rng.sample(range(len(deck)), num_cards_masked))
        visible_cards = torch.zeros(self.num_cards)
        missing_cards = torch.zeros(self.num_cards)
        for position, card in enumerate(deck):
            if position in masked_cards_positions:
                missing_cards[card] = 1.0
            else:
                visible_cards[card] = 1.0

        return visible_cards, missing_cards, self.weights[index]


class DeckBuilderModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)

        return x