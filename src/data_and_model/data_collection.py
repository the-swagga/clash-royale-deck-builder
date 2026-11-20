from api_call_scripts.get_player_battle_log_json import fetch_process_save_battle_log
from api_call_scripts.get_top_1k_player_tags_json import save_top_1k_players

# BATTLE LOG NOT FILTERED FOR ONLY RANKED FIX IMMEDIATELY

save_top_1k_players("../../data/raw_data/top_1k_players.json")

fetch_process_save_battle_log("api_call_scripts/api_key.txt",
                              "../../data/raw_data/top_1k_players.json",
                              "../../data/raw_data/top_1k_battle_log_nov_26_.json")