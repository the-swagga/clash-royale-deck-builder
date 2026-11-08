import urllib.request as r
import json
import numpy as np
from api_call_scripts.modular_functions import save_to_json

BASE = "https://api.clashroyale.com/v1"

def load_api_key(path="api_key.txt"):
    with open(path, "r") as f:
        return f.readline()


def read_top_1k_players_tags(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        players_json = json.load(f)

    player_tags = []
    for player in players_json:
        player_tags.append(player['tag'])

    return player_tags


def fetch_battle_log(key, player_tag):
    endpoint = f"/players/%23{player_tag}/battlelog"
    request = r.Request(BASE + endpoint, None, {"Authorization": "Bearer %s" % key})
    response = r.urlopen(request).read().decode("utf-8")

    return json.loads(response)


def parse_battle(battle):
    player_name = battle['team'][0]['name']
    print(f"Player Name: {player_name}")

    player_rank = battle['team'][0].get('globalRank')
    player_ranked = player_rank is not None
    if player_ranked:
        print(f"Rank: {player_rank}")

    opponent_rank = battle['opponent'][0].get('globalRank')
    opponent_ranked = opponent_rank is not None
    if opponent_ranked:
        print(f"Opponent Rank: {opponent_rank}")

    cards = []
    for card in battle['team'][0]['cards']:
        card_name = ""
        evo_hero_indicator = card.get('evolutionLevel', 0)
        if evo_hero_indicator == 1:
            card_name = f"Evo {card['name']}"
        elif evo_hero_indicator == 2:
            card_name = f"Hero {card['name']}"
        elif evo_hero_indicator == 0:
            card_name = card['name']
        else:
            print("ERROR: Invalid evolution level.")

        cards.append(card_name)

    if len(cards) == 8:
        print(f"Deck: {cards[0]}, {cards[1]}, {cards[2]}, {cards[3]}, {cards[4]}, {cards[5]}, {cards[6]}, {cards[7]}")
    else:
        print("WARNING: Deck does not have 8 cards.")

    player_crowns = battle['team'][0]['crowns']
    opponent_crowns = battle['opponent'][0]['crowns']
    if player_crowns > opponent_crowns:
        player_won = True
    elif player_crowns < opponent_crowns:
        player_won = False
    else:
        player_won = False
    print(f"Player Won: {player_won}")

    player_tower_hp = 0
    player_princess_tower_hp = battle['team'][0].get('princessTowersHitPoints')
    if player_princess_tower_hp is not None:
        for tower_hp in battle['team'][0]['princessTowersHitPoints']:
            player_tower_hp += tower_hp
    player_king_tower_hp = battle['team'][0].get('kingTowersHitPoints', 0)
    player_tower_hp += player_king_tower_hp
    opponent_tower_hp = 0
    opponent_princess_tower_hp = battle['opponent'][0].get('princessTowersHitPoints')
    if opponent_princess_tower_hp is not None:
        for tower_hp in battle['opponent'][0]['princessTowersHitPoints']:
            opponent_tower_hp += tower_hp
    opponent_king_tower_hp = battle['opponent'][0].get('kingTowersHitPoints', 0)
    opponent_tower_hp += opponent_king_tower_hp

    tower_hp_diff = player_tower_hp - opponent_tower_hp
    crown_diff = player_crowns - opponent_crowns
    print(f"Victory Margin: {crown_diff} towers and {tower_hp_diff} HP")

    if player_ranked and opponent_ranked:
        log_rank_diff = np.log(player_rank) - np.log(opponent_rank)
    else:
        log_rank_diff = None

    parsed_battle = {
        "Player Name": player_name,
        "Rank Diff": log_rank_diff,
        "Deck": cards,
        "Player Won": player_won,
        "Crown Diff": crown_diff,
        "Tower HP Diff": tower_hp_diff
    }

    print()

    return parsed_battle


def fetch_process_save_battle_log(api_key_path, top_1k_players_path, data_directory_path):
    key = load_api_key(api_key_path)

    top_1k_players_tags = read_top_1k_players_tags(top_1k_players_path)

    all_players_battle_records = []
    for player_tag in top_1k_players_tags:
        battle_log = fetch_battle_log(key, player_tag)
        player_battle_records = []
        for battle in battle_log:
            parsed_battle = parse_battle(battle)
            player_battle_records.append(parsed_battle)
        all_players_battle_records.append(player_battle_records)

    save_to_json(data_directory_path, all_players_battle_records)


