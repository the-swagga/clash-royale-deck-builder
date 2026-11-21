import urllib.request as r
import json
from src.data_and_model.api_call_scripts.modular_functions import get_base_url, load_api_key

BASE = get_base_url()

def save_card_icons():
    icon_urls = get_card_icon_urls()
    for card, url in icon_urls.items():
        r.urlretrieve(url, f"../../../data/icons/{card}.png")


def get_card_icon_urls():
    key = load_api_key()
    endpoint = "/cards"
    request = r.Request(BASE + endpoint, None, {"Authorization": "Bearer %s" % key})
    response = r.urlopen(request).read().decode("utf-8")
    cards_json = json.loads(response)

    icon_urls = {}
    for card in cards_json["items"]:
        card_name = card["name"]
        if card["iconUrls"].get("evolutionMedium"):
            icon_urls[f"Evo {card_name}"] = card["iconUrls"]["evolutionMedium"]
        if card["iconUrls"].get("heroMedium"):
            icon_urls[f"Hero {card_name}"] = card["iconUrls"]["heroMedium"]
        icon_urls[card_name] = card["iconUrls"]["medium"]

    return icon_urls

save_card_icons()