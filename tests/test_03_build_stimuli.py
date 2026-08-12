import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

build_stimuli = import_module("03_build_stimuli")


def test_build_items_assigns_ids_in_list_order():
    items = build_stimuli.build_items()
    ids = [item["item_id"] for item in items]

    assert len(items) == 100
    assert ids[0] == "C001"
    assert ids[39] == "C040"
    assert ids[40] == "D001"
    assert ids[74] == "D035"
    assert ids[75] == "F001"
    assert ids[99] == "F025"


def test_build_items_ids_are_unique():
    items = build_stimuli.build_items()
    ids = [item["item_id"] for item in items]

    assert len(set(ids)) == len(ids)


def test_build_orders_stable_across_runs():
    orders_a = build_stimuli.build_orders(100)
    orders_b = build_stimuli.build_orders(100)

    assert orders_a == orders_b


def test_build_orders_stable_across_process_boundary():
    # A fresh Random(seed) per call must not be perturbed by unrelated
    # random usage elsewhere in the process.
    import random
    random.random()  # perturb the global RNG state, which build_orders must not depend on

    orders_a = build_stimuli.build_orders(100)
    orders_b = build_stimuli.build_orders(100)

    assert orders_a == orders_b


def test_all_items_appear_exactly_once_per_pass():
    items = build_stimuli.build_items()
    orders = build_stimuli.build_orders(len(items))

    for pass_name, idx_list in orders.items():
        assert sorted(idx_list) == list(range(len(items))), pass_name


def test_passes_present_and_named():
    orders = build_stimuli.build_orders(100)

    assert set(orders.keys()) == {"A_modal", "B_natural", "C_creak"}


def test_orders_differ_between_passes():
    orders = build_stimuli.build_orders(100)

    assert orders["A_modal"] != orders["B_natural"]
    assert orders["B_natural"] != orders["C_creak"]
    assert orders["A_modal"] != orders["C_creak"]


def test_end_to_end_writes_expected_csv_rows(tmp_path):
    items_path = tmp_path / "items.csv"
    orders_path = tmp_path / "orders.csv"

    items = build_stimuli.build_items()
    orders = build_stimuli.build_orders(len(items))
    build_stimuli.write_items_csv(items, items_path)
    build_stimuli.write_orders_csv(items, orders, orders_path)

    import csv

    with items_path.open(encoding="utf-8") as f:
        item_rows = list(csv.DictReader(f))
    with orders_path.open(encoding="utf-8") as f:
        order_rows = list(csv.DictReader(f))

    assert len(item_rows) == 100
    assert {row["item_id"] for row in item_rows} == {item["item_id"] for item in items}
    assert len(order_rows) == 300

    for pass_name in build_stimuli.PASS_NAMES:
        pass_item_ids = [row["item_id"] for row in order_rows if row["pass"] == pass_name]
        assert len(pass_item_ids) == 100
        assert set(pass_item_ids) == {item["item_id"] for item in items}
