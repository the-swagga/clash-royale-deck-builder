import json
import torch
from model.model import DeckBuilderModel

MAX_ATTEMPTS = 1000
MIN_DECKS_TO_GENERATE = 1
MAX_DECKS_TO_GENERATE = 10

input_card_names = ["Evo Goblin Barrel"]
excluded_card_names = []
for card in input_card_names:
    if card in excluded_card_names:
        raise Exception(f"{card} in input cards and excluded cards.")

with open("processed_data/card_ids.json", encoding="utf-8") as f:
    card_ids  = json.load(f)

card_names = {}
for card, id in card_ids.items():
    card_names[id] = card

model = DeckBuilderModel(len(card_ids))
model.load_state_dict(torch.load("saved_models/deck_builder_model.pkl"))
model.eval()

with open("deck_builder_rules/banned_pairs.json", encoding="utf-8") as f:
    banned_pairs = json.load(f)

wincon_names = set(banned_pairs["Banned Wincon Pairs"]["All Wincons"])
wincon_ids = []
for name in wincon_names:
    wincon_ids.append(card_ids[name])

used_wincons = {}
for name in wincon_names:
    used_wincons[card_ids[name]] = False

def output(num_decks_to_generate):
    if num_decks_to_generate < MIN_DECKS_TO_GENERATE:
        num_decks_to_generate = MIN_DECKS_TO_GENERATE
    elif num_decks_to_generate > MAX_DECKS_TO_GENERATE:
        num_decks_to_generate = MAX_DECKS_TO_GENERATE

    unique_decks_generated = []
    num_unique_decks_generated = len(unique_decks_generated)
    attempts = 0
    while num_unique_decks_generated < num_decks_to_generate and attempts < MAX_ATTEMPTS:
        attempts += 1

        out_deck = input_card_names.copy()

        picked_wincons = []
        has_wincon = False
        for card in input_card_names:
            if card_ids[card] in used_wincons:
                used_wincons[card_ids[card]] = True
                has_wincon = True

        while len(out_deck) < 8:
            visible = torch.zeros(len(card_ids))
            for card in out_deck:
                visible[card_ids[card]] = 1.0
            visible = visible.unsqueeze(0)

            with torch.no_grad():
                logits = model(visible)
            logits = logits.masked_fill(visible.bool(), float("-inf"))
            for wincon_id, used in used_wincons.items():
                if used:
                    logits[0, wincon_id] = float("-inf")
            for card in excluded_card_names:
                logits[0, card_ids[card]] = float("-inf")
            probabilities = torch.softmax(logits, dim=-1)

            top_probabilities, top_indices = torch.topk(probabilities, int(len(card_ids) * 0.2), dim=-1)
            final_probabilities = torch.zeros_like(probabilities)
            final_probabilities.scatter_(1, top_indices, top_probabilities)
            final_probabilities = final_probabilities / final_probabilities.sum()

            prediction = torch.multinomial(final_probabilities, 1).item()

            if prediction in used_wincons:
                picked_wincons.append(prediction)
                has_wincon = True

            out_deck.append(card_names[prediction])

        deck_already_generated = False
        for deck in unique_decks_generated:
            if identical_decks_check(deck, out_deck):
                deck_already_generated = True
                break

        if has_wincon and not deck_already_generated:
            for id in picked_wincons:
                used_wincons[id] = True
            sorted_out_deck = out_deck.copy()
            sorted_out_deck.sort()
            for card in sorted_out_deck:
                print(card)
            print()
            unique_decks_generated.append(out_deck)

        num_unique_decks_generated = len(unique_decks_generated)

def identical_decks_check(deck_1, deck_2):
    deck_1.sort()
    deck_2.sort()
    if deck_1 == deck_2:
        return True
    else:
        return False

output(num_decks_to_generate=5)