import math
import json
import random
import glob
from src.data_and_model.api_call_scripts.modular_functions import save_to_json

TOWER_HP_SCALE = 1000

def calc_record_weight(rank_diff, crown_diff, tower_hp_diff):
    rank_diff_bonus = 0.3 * rank_diff
    weight = (1.0 * crown_diff) + ((0.3 * tower_hp_diff) / TOWER_HP_SCALE)
    weight = weight + rank_diff_bonus
    sigmoid = 1 / (1 + math.exp(-weight))
    out_weight = 0.2 + (sigmoid * 0.8)

    return out_weight

random.seed(38)

data_files = sorted(glob.glob("../../../data/raw_data/top_1k_battle_log_*.json"))

data = []
for path in data_files:
    with open(path, encoding="utf-8") as f:
        data.append(json.load(f))

records = []
unique_records = set()
for file in data:
    for battle_log in file:
        player_key = battle_log[0]["Player Name"]
        for record in battle_log:
            if record["Rank Diff"] is None:
                continue

            record_weight = calc_record_weight(record["Rank Diff"],
                                               record["Crown Diff"],
                                               record["Tower HP Diff"])

            unique_identifier = (player_key, record_weight)
            if unique_identifier in unique_records:
                continue

            if len(record["Deck"]) == 8:
                unique_records.add(unique_identifier)
                records.append({"player": player_key,
                                "deck": record["Deck"],
                                "weight": record_weight
                                })
print(f"Total records: {len(unique_records)}")

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
for i, card in enumerate(sorted(all_cards)):
    card_ids[card] = i

for record in training_records + validation_records:
    processed_deck = []
    for card in record["deck"]:
        processed_deck.append(card_ids[card])
    record["deck"] = processed_deck

save_to_json("../../../data/processed_data/card_ids.json", card_ids)
save_to_json("../../../data/processed_data/training_records.json", training_records)
save_to_json("../../../data/processed_data/validation_records.json", validation_records)