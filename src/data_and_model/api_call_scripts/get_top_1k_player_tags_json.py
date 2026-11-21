import urllib.request as r
import json
from src.data_and_model.api_call_scripts.modular_functions import get_base_url, load_api_key, save_to_json

BASE = get_base_url()

def save_top_1k_players(file_path):
    key = load_api_key()
    endpoint = "/locations/global/pathoflegend/players"
    request = r.Request(BASE + endpoint, None, {"Authorization": "Bearer %s" % key})
    response = r.urlopen(request).read().decode("utf-8")
    players_json = json.loads(response)

    players_tag_name_rank = []
    for player in players_json["items"]:
        player_dict = {}
        modified_tag = player["tag"].replace('#', '')
        player_dict["tag"] = modified_tag
        player_dict["name"] = player["name"]
        player_dict["rank"] = player["rank"]
        players_tag_name_rank.append(player_dict)

    save_to_json(file_path, players_tag_name_rank)