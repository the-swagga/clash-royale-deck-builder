import json
import torch
from model.model import DeckBuilderModel


# --- Constants --- #
MAX_ATTEMPTS = 1000
MIN_DECKS_TO_GENERATE = 1
MAX_DECKS_TO_GENERATE = 10
DECK_SIZE = 8
TOP_K_SAMPLE_PERCENTAGE = 0.2


# --- Prompt --- #
input_card_names = ["Evo Goblin Barrel"]
excluded_card_names = []
for card in input_card_names:
    if card in excluded_card_names:
        raise Exception(f"{card} in input cards and excluded cards.")


# --- Load data and model --- #
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

used_wincons_names = {}
for name in wincon_names:
    used_wincons_names[card_ids[name]] = False


# --- Helper functions --- #
def identical_decks_check(deck_1, deck_2): # Check if two generated decks are identical
    deck_1.sort()
    deck_2.sort()
    if deck_1 == deck_2:
        return True
    else:
        return False


def generate_next_card(out_deck, excluded_cards): # Predict the next card given an incomplete deck
    visible = torch.zeros(len(card_ids))
    for card in out_deck:
        visible[card_ids[card]] = 1.0
    visible = visible.unsqueeze(0)

    with torch.no_grad():
        logits = model(visible)
    logits = logits.masked_fill(visible.bool(), float("-inf"))
    for wincon_id, used in used_wincons_names.items():
        if used:
            logits[0, wincon_id] = float("-inf")
    for card in excluded_cards:
        logits[0, card_ids[card]] = float("-inf")
    probabilities = torch.softmax(logits, dim=-1)

    top_probabilities, top_indices = torch.topk(probabilities, int(len(card_ids) * 0.2), dim=-1)
    final_probabilities = torch.zeros_like(probabilities)
    final_probabilities.scatter_(1, top_indices, top_probabilities)
    final_probabilities = final_probabilities / final_probabilities.sum()
    prediction = torch.multinomial(final_probabilities, 1).item()

    return prediction


def generate_deck(input_cards, excluded_cards, used_wincons): # Generate a deck given an incomplete input deck and a list of excluded cards
    out_deck = input_cards.copy()
    picked_wincons = []
    has_wincon = False
    for card in input_card_names:
        if card_ids[card] in used_wincons:
            used_wincons[card_ids[card]] = True
            has_wincon = True

    while len(out_deck) < DECK_SIZE:
        prediction = generate_next_card(out_deck, excluded_cards)

        if prediction in used_wincons:
            picked_wincons.append(prediction)
            has_wincon = True

        out_deck.append(card_names[prediction])

    return out_deck, picked_wincons, has_wincon


# --- Main function --- #
def output(num_decks_to_generate, input_cards, excluded_cards, used_wincons): # Generate a specified amount of decks
    if num_decks_to_generate < MIN_DECKS_TO_GENERATE:
        num_decks_to_generate = MIN_DECKS_TO_GENERATE
    elif num_decks_to_generate > MAX_DECKS_TO_GENERATE:
        num_decks_to_generate = MAX_DECKS_TO_GENERATE

    unique_decks_generated = []
    attempts = 0
    while len(unique_decks_generated) < num_decks_to_generate and attempts < MAX_ATTEMPTS:
        attempts += 1

        out_deck, picked_wincons, has_wincon = generate_deck(input_cards, excluded_cards, used_wincons)

        deck_already_generated = False
        for deck in unique_decks_generated:
            if identical_decks_check(deck, out_deck):
                deck_already_generated = True
                break

        if has_wincon and not deck_already_generated:
            for id in picked_wincons:
                used_wincons_names[id] = True
            sorted_out_deck = out_deck.copy()
            sorted_out_deck.sort()
            for card in sorted_out_deck:
                print(card)
            print()
            unique_decks_generated.append(out_deck)


output(num_decks_to_generate=5, input_cards=input_card_names, excluded_cards=excluded_card_names, used_wincons=used_wincons_names)