import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DeckBuilderDataset, DeckBuilderModel

# --- Training Variables --- #
TRAINING_SET_PATH = "../../../data/processed_data/training_records.json"
VALIDATION_SET_PATH = "../../../data/processed_data/validation_records.json"
CARD_IDS_PATH = "../../../data/processed_data/card_ids.json"
BATCH_SIZE = 128
EPOCH_NUM = 128
LEARN_RATE = 0.005
PATIENCE = 15

def train_model(seed=38, show_loss=True):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CARD_IDS_PATH, encoding="utf-8") as f:
        card_ids = json.load(f)
        num_cards_seen = len(card_ids)

    training_set = DeckBuilderDataset(TRAINING_SET_PATH, num_cards_seen, is_train=True)
    validation_set = DeckBuilderDataset(VALIDATION_SET_PATH, num_cards_seen, is_train=False, seed=seed)

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE)

    model = DeckBuilderModel(num_cards_seen).to(device)
    save_path = "../../../saved_models/main_model.pkl"

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    best_validation_loss = float("inf")
    patience_count = 0

    for epoch in range(EPOCH_NUM):
        model.train()
        training_loss = 0

        for visible_batch, missing_batch, weight_batch in training_loader:
            visible_batch = visible_batch.to(device)
            missing_batch = missing_batch.to(device)
            weight_batch = weight_batch.to(device)

            optimiser.zero_grad()
            predictions = model(visible_batch)
            loss_per_sample = loss_function(predictions, missing_batch).sum(dim=1)
            loss = (loss_per_sample * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            optimiser.step()
            training_loss += loss.item()

        model.eval()
        weighted_validation_loss = 0
        unweighted_validation_loss = 0
        weight_total = 0

        with torch.no_grad():
            for visible_batch, missing_batch, weight_batch in validation_loader:
                visible_batch = visible_batch.to(device)
                missing_batch = missing_batch.to(device)
                weight_batch = weight_batch.to(device)

                predictions = model(visible_batch)
                loss_per_sample = loss_function(predictions, missing_batch).sum(dim=1)

                weighted_validation_loss += (loss_per_sample * weight_batch).sum().item()
                weight_total += weight_batch.sum().item()
                unweighted_validation_loss += loss_per_sample.sum().item()

        weighted_validation_loss /= weight_total
        unweighted_validation_loss /= len(validation_set)

        if show_loss:
            print(f"Epoch {epoch + 1}/{EPOCH_NUM} - Train Loss: {training_loss / len(training_loader):.4f} - Val Loss (weighted): {weighted_validation_loss:.4f} - Val Loss (unweighted): {unweighted_validation_loss:.4f}")

        if unweighted_validation_loss < best_validation_loss:
            best_validation_loss = unweighted_validation_loss
            torch.save(model.state_dict(), save_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count > PATIENCE:
                break

    total_parameters = 0
    for parameter in model.parameters():
        total_parameters += parameter.numel()
    print(f"Total parameters: {total_parameters}")


if __name__ == "__main__":
    train_model()