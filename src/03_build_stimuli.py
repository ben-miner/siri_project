"""Regenerate the stimulus item list and read orders deterministically.

Item IDs are assigned in list order: C001-C040 (COMMANDS), D001-D035
(DECLARATIVES), F001-F025 (FINAL_FALL). The master list is the
concatenation COMMANDS + DECLARATIVES + FINAL_FALL, giving indices 0-99.

Three read orders are produced by shuffling that index list with
random.Random(20270607 + k), k=0,1,2, mapping to passes A_modal,
B_natural, and C_creak respectively. Because the seed and the master
list order are fixed, re-running this script reproduces byte-identical
items.csv and orders.csv.
"""

import argparse
import csv
import random
from pathlib import Path

COMMANDS = [
    "Set a timer for twelve minutes.",
    "Text Marcus that I'm running late.",
    "What's the weather in Newton tomorrow?",
    "Play the new Noah Kahan album.",
    "Remind me to email my advisor on Monday.",
    "How long is the flight to Tokyo?",
    "Add oat milk to my grocery list.",
    "Call the gelato shop on Orchard Street.",
    "Turn off the bedroom lights.",
    "What time does the library close today?",
    "Navigate to Grand Central Terminal.",
    "Send a photo to the choir group chat.",
    "Show me my calendar for next Thursday.",
    "Convert eighty degrees Fahrenheit to Celsius.",
    "Skip to the next song.",
    "What's the score of the Mets game?",
    "Set an alarm for six forty-five.",
    "Find a ramen place open past midnight.",
    "Read my last message from Anjali.",
    "How do you say thank you in Japanese?",
    "Start a workout on my watch.",
    "Cancel my reminder about the dentist.",
    "Take a note about the pitch analysis.",
    "What is twenty percent of forty-three dollars?",
    "Turn the volume down a little.",
    "Open the Praat documentation in Safari.",
    "Schedule a meeting for Friday afternoon.",
    "Is it going to rain this evening?",
    "Play something quiet while I study.",
    "Delete the last voice memo.",
    "What's the fastest way to Morningside Heights?",
    "Add a wisteria trellis to my shopping list.",
    "Tell me a joke about linguists.",
    "Show me photos from Cape Cod.",
    "How many days until September first?",
    "Turn on do not disturb until noon.",
    "Where is the nearest pharmacy?",
    "Email the lab about the annotation deadline.",
    "What's on my agenda this morning?",
    "Play white noise for thirty minutes.",
]

DECLARATIVES = [
    "The old bridge groaned under the weight of the truck.",
    "She poured the vinegar into a shallow glass dish.",
    "Seven zebras wandered across the dusty road.",
    "My grandmother kept her thimbles in a wooden box.",
    "The judge asked for silence in the crowded room.",
    "Thick fog settled over the harbor before dawn.",
    "He measured the flour twice and still got it wrong.",
    "Purple thistles grew along the railway embankment.",
    "The choir rehearsed the same passage eleven times.",
    "A single moth circled the porch light all evening.",
    "They shipped the equipment in reinforced crates.",
    "The tide pulled the small boat toward the rocks.",
    "Cinnamon and cloves filled the kitchen with warmth.",
    "Nobody remembered who had locked the back door.",
    "The librarian stamped each card with quiet precision.",
    "Wet leaves clogged the gutter above the window.",
    "I bought a secondhand bicycle for forty dollars.",
    "The mountain village looked empty in the photograph.",
    "Three sparrows fought over a crust of bread.",
    "His youngest brother plays trumpet in the marching band.",
    "The recipe called for saffron, which we could not find.",
    "Snow buried the fence posts by early January.",
    "She wrote the whole essay on a borrowed laptop.",
    "The engine coughed once and then went silent.",
    "Amber light spilled across the polished floorboards.",
    "We watched the storm move east across the valley.",
    "The tailor pinned the hem with practiced fingers.",
    "Salt crusted the windows of the beach house.",
    "Their apartment smelled faintly of turpentine.",
    "A heron stood motionless at the edge of the pond.",
    "The clock in the hallway runs about four minutes fast.",
    "He folded the newspaper and set it on the table.",
    "Bright green algae covered the surface of the water.",
    "The children traded stickers during the long bus ride.",
    "Rain hammered the tin roof for most of the night.",
]

FINAL_FALL = [
    "I don't really know what she was thinking, honestly.",
    "We ended up walking the whole way home, anyway.",
    "It was one of those afternoons where nothing happened at all.",
    "He said he'd call back, but he never really did.",
    "The whole thing fell apart before anyone noticed, apparently.",
    "I guess that's just how it goes sometimes.",
    "She left the party early and nobody said anything.",
    "It's not that I mind, it's just kind of strange.",
    "They moved out to Oregon a couple of years ago.",
    "Nothing about that story adds up, if you think about it.",
    "I'd been meaning to write back for a while now.",
    "We sat there in the car for maybe twenty minutes.",
    "He's been doing that same job for eleven years.",
    "It rained the entire week we were on the island.",
    "I never learned how to swim, which is embarrassing.",
    "That's pretty much all anyone talked about afterward.",
    "She kept apologizing even though it wasn't her fault.",
    "The apartment was smaller than we expected, obviously.",
    "I think I left my keys somewhere in the stairwell.",
    "It turned out fine, more or less, in the end.",
    "Nobody warned us about the traffic on the way in.",
    "He mentioned it once and then never brought it up again.",
    "We were both too tired to argue about it anymore.",
    "I've been putting that conversation off for months.",
    "It felt like the longest week of the entire summer.",
]

ITEM_TYPES = [
    ("C", COMMANDS, "command", 40),
    ("D", DECLARATIVES, "declarative", 35),
    ("F", FINAL_FALL, "final_fall", 25),
]

PASS_NAMES = ["A_modal", "B_natural", "C_creak"]
SEED_BASE = 20270607


def build_items() -> list[dict]:
    for prefix, sentences, item_type, expected_n in ITEM_TYPES:
        if len(sentences) != expected_n:
            raise ValueError(
                f"{item_type} list has {len(sentences)} sentences, expected {expected_n}"
            )

    items = []
    for prefix, sentences, item_type, _ in ITEM_TYPES:
        for i, text in enumerate(sentences, start=1):
            items.append({
                "item_id": f"{prefix}{i:03d}",
                "item_type": item_type,
                "text": text,
            })

    ids = [item["item_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item_id in generated item list")

    return items


def build_orders(n_items: int) -> dict:
    orders = {}
    for k, pass_name in enumerate(PASS_NAMES):
        idx = list(range(n_items))
        random.Random(SEED_BASE + k).shuffle(idx)
        orders[pass_name] = idx
    return orders


def write_items_csv(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "item_type", "text"])
        writer.writeheader()
        writer.writerows(items)


def write_orders_csv(items: list[dict], orders: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pass", "position", "item_id"])
        writer.writeheader()
        for pass_name in PASS_NAMES:
            for position, idx in enumerate(orders[pass_name], start=1):
                writer.writerow({
                    "pass": pass_name,
                    "position": position,
                    "item_id": items[idx]["item_id"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items-out", type=Path, default=Path("stimuli/items.csv"),
        help="Output path for the item list (default: stimuli/items.csv)",
    )
    parser.add_argument(
        "--orders-out", type=Path, default=Path("stimuli/orders.csv"),
        help="Output path for the read orders (default: stimuli/orders.csv)",
    )
    args = parser.parse_args()

    items = build_items()
    orders = build_orders(len(items))

    write_items_csv(items, args.items_out)
    write_orders_csv(items, orders, args.orders_out)

    print(
        f"Wrote {len(items)} items to {args.items_out} and "
        f"{len(orders) * len(items)} order rows ({len(orders)} passes x {len(items)} items) "
        f"to {args.orders_out}"
    )


if __name__ == "__main__":
    main()
