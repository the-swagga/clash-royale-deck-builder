import json
import torch
from model.model import DeckBuilderModel


# --- Constants --- #
MIN_DECKS_TO_GENERATE = 1
MAX_DECKS_TO_GENERATE = 10
MAX_CARD_ATTEMPTS = MAX_DECKS_TO_GENERATE * 10
MAX_DECK_ATTEMPTS = MAX_DECKS_TO_GENERATE * 10
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

with open("deck_builder_rules/wincons.json", encoding="utf-8") as f:
    wincons = json.load(f)

wincon_names = set(wincons["All Wincons"])
wincon_ids = set()
for name in wincon_names:
    wincon_ids.add(card_ids[name])

used_wincons_names = {}
for name in wincons["Hard Wincons"]:
    used_wincons_names[card_ids[name]] = False


# --- Misc helper functions --- #
def get_card_type(card):
    if card in ["Archer Queen", "Boss Bandit", "Goblinstein", "Golden Knight",
                "Little Prince", "Mighty Miner", "Monk","Skeleton King"]:
        return "Champion"
    if card[:3] == "Evo":
        return "Evo"
    if card[:4] == "Hero":
        return "Hero"

    return "Base"


def identical_base_card_check(card1, card2):
    card1_type = get_card_type(card1)
    card2_type = get_card_type(card2)
    accepted_types = ["Evo", "Hero"]
    if card1_type not in accepted_types and card2_type not in accepted_types:
        return card1 == card2

    if card1_type == "Evo":
        stripped_card1 = card1.removeprefix("Evo ")
    elif card1_type == "Hero":
        stripped_card1 = card1.removeprefix("Hero ")
    else:
        stripped_card1 = card1

    if card2_type == "Evo":
        stripped_card2 = card2.removeprefix("Evo ")
    elif card2_type == "Hero":
        stripped_card2 = card2.removeprefix("Hero ")
    else:
        stripped_card2 = card2

    if stripped_card1 == stripped_card2:
        return True
    else:
        return False


def identical_decks_check(deck_1, deck_2):
    deck_1.sort()
    deck_2.sort()
    if deck_1 == deck_2:
        return True
    else:
        return False


def get_evo_champion_hero_count(deck):
    evo_count = 0
    champion_hero_count = 0

    for card in deck:
        card_type = get_card_type(card)
        if card_type == "Evo":
            evo_count += 1
        elif card_type == "Hero" or card_type == "Champion":
            champion_hero_count += 1

    total_count = evo_count + champion_hero_count

    return evo_count, champion_hero_count, total_count


def is_eligible_card_type(deck, card):
    card_type = get_card_type(card)
    if card_type == "Base":
        return True

    evo_count, champion_hero_count, total_count = get_evo_champion_hero_count(deck)

    if total_count == 3:
        return False
    if card_type == "Evo" and evo_count == 2:
        return False
    if card_type == "Hero" and champion_hero_count == 2:
        return False
    if card_type == "Champion" and champion_hero_count == 2:
        return False

    return True


# --- Main helper functions --- #
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

    if not is_eligible_card_type(out_deck, card_names[prediction]):
        return None

    return prediction


def generate_deck(input_cards, excluded_cards, used_wincons): # Generate a deck given an incomplete input deck and a list of excluded cards
    out_deck = input_cards.copy()
    picked_wincons = []
    has_wincon = False
    for card in input_cards:
        if card_ids[card] in wincon_ids:
            has_wincon = True
        if card_ids[card] in used_wincons:
            used_wincons[card_ids[card]] = True

    attempts = 0
    while len(out_deck) < DECK_SIZE and attempts < MAX_CARD_ATTEMPTS:
        attempts += 1

        prediction = generate_next_card(out_deck, excluded_cards)
        if prediction is None:
            continue

        is_duplicate = False
        for card in out_deck:
            is_duplicate = identical_base_card_check(card_names[prediction], card)
            if is_duplicate:
                break
        if is_duplicate:
            continue

        if prediction in wincon_ids:
            has_wincon = True
        if prediction in used_wincons:
            picked_wincons.append(prediction)

        out_deck.append(card_names[prediction])

    while len(out_deck) < DECK_SIZE:
        out_deck.append("") # For cases where an incomplete deck was generated (ie max attempts exceeded)

    return out_deck, picked_wincons, has_wincon


# --- Main function --- #
def output(num_decks_to_generate, input_cards, excluded_cards, used_wincons): # Generate a specified amount of decks
    if num_decks_to_generate < MIN_DECKS_TO_GENERATE:
        num_decks_to_generate = MIN_DECKS_TO_GENERATE
    elif num_decks_to_generate > MAX_DECKS_TO_GENERATE:
        num_decks_to_generate = MAX_DECKS_TO_GENERATE

    unique_decks_generated = []
    attempts = 0
    while len(unique_decks_generated) < num_decks_to_generate and attempts < MAX_DECK_ATTEMPTS:
        attempts += 1

        out_deck, picked_wincons, has_wincon = generate_deck(input_cards, excluded_cards, used_wincons)
        if "" in out_deck:
            continue

        deck_already_generated = False
        for deck in unique_decks_generated:
            if identical_decks_check(deck, out_deck):
                deck_already_generated = True
                break

        #if has_wincon and not deck_already_generated:
        if not deck_already_generated:
            for id in picked_wincons:
                used_wincons_names[id] = True
            sorted_out_deck = out_deck.copy()
            sorted_out_deck.sort()
            for card in sorted_out_deck:
                print(card)
            print()
            unique_decks_generated.append(out_deck)


output(num_decks_to_generate=10, input_cards=input_card_names, excluded_cards=excluded_card_names, used_wincons=used_wincons_names)