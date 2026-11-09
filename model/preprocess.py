import math
import json
import random
import glob
from api_call_scripts.modular_functions import save_to_json

tower_hp_scale = 1000

def calc_record_weight(rank_diff, crown_diff, tower_hp_diff):
    rank_diff_bonus = 0.3 * rank_diff
    weight = (1.0 * crown_diff) + ((0.3 * tower_hp_diff) / tower_hp_scale)
    weight = weight + rank_diff_bonus
    sigmoid = 1 / (1 + math.exp(-weight))
    out_weight = 0.2 + (sigmoid * 0.8)

    return out_weight

random.seed(38)

data_files = sorted(glob.glob("../raw_data/top_1k_battle_log_*.json"))

data = []
for path in data_files:
    with open(path, encoding="utf-8") as f:
        data.append(json.load(f))

records = []
for file in data:
    for battle_log in file:
        player_key = battle_log[0]["Player Name"]
        for record in battle_log:
            if record["Rank Diff"] is None:
                continue
            if len(record["Deck"]) == 8:
                records.append({"player": player_key,
                                "deck": record["Deck"],
                                "weight": calc_record_weight(record["Rank Diff"],
                                                             record["Crown Diff"],
                                                             record["Tower HP Diff"])
                                })

unique_players = set()
for record in records:
    unique_players.add(record["player"])
player_keys = sorted(unique_players)
random.shuffle(player_keys)

validation_split = 0.1
validation_split = int(len(player_keys) * validation_split)
validation_keys = set(player_keys[:validation_split])

training_records = []
validation_records = []
for record in records:
    if record["player"] in validation_keys:
        validation_records.append(record)
    else:
        training_records.append(record)

all_cards = set()
for record in training_records + validation_records:
    for card in record["deck"]:
        all_cards.add(card)

card_ids = {}
for i, card in enumerate(sorted(all_cards)):#
    card_ids[card] = i

for record in training_records + validation_records:
    processed_deck = []
    for card in record["deck"]:
        processed_deck.append(card_ids[card])
    record["deck"] = processed_deck

save_to_json("../processed_data/card_ids.json", card_ids)
save_to_json("../processed_data/training_records.json", training_records)
save_to_json("../processed_data/validation_records.json", validation_records)