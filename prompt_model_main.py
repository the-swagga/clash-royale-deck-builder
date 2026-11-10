import json
import torch

from model.model import DeckBuilderModel

BANNED_PAIRS = {
    "Hog Rider": ["Ram Rider", "Evo Battle Ram", "Battle Ram"]
}

input_card_names = ["Evo Royal Hogs", "Phoenix", "Dark Prince", "Vines"]

with open("processed_data/card_ids.json", encoding="utf-8") as f:
    card_ids  = json.load(f)

card_names = {}
for card, id in card_ids.items():
    card_names[id] = card

model = DeckBuilderModel(len(card_ids))
model.load_state_dict(torch.load("saved_models/deck_builder_model.pkl"))
model.eval()


def output():
    for card_name in input_card_names:
        print(card_name)

    predictions = []
    out_deck = input_card_names
    while len(out_deck) < 8:
        input_card_ids = []
        for card in input_card_names:
            input_card_ids.append(card_ids[card])

        visible = torch.zeros(len(card_ids))
        for id in input_card_ids:
            visible[id] = 1.0

        visible = visible.unsqueeze(0)

        with torch.no_grad():
            logits = model(visible)
        logits = logits.masked_fill(visible.bool(), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)

        top_probabilities, top_indices = torch.topk(probabilities, 15, dim=-1)
        final_probabilities = torch.zeros_like(probabilities)
        final_probabilities.scatter_(1, top_indices, top_probabilities)
        final_probabilities = final_probabilities / final_probabilities.sum()

        prediction = torch.multinomial(final_probabilities, 1).item()
        print(f"{card_names[prediction]}: {probabilities[0, prediction].item():.2%}")

        out_deck.append(card_names[prediction])

output()