"""language_only_task_info reads pick/place/prompts from the instruction alone (LIBERO wording)."""
import pytest

from spark_bench.libero_pro.bddl import language_only_task_info

CASES = [
    ("pick up the alphabet soup and place it in the basket", "alphabet soup", "basket"),
    ("Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "akita black bowl", "plate"),
    ("pick up the black bowl between the plate and the ramekin and place it on the plate", "black bowl", "plate"),
    ("put the bowl on the plate", "bowl", "plate"),
    ("put the wine bottle on top of the cabinet", "wine bottle", "cabinet"),
    ("push the plate to the front of the stove", "plate", "stove"),
    ("open the middle drawer of the cabinet", "middle drawer", "none"),
    ("turn on the stove", "stove", "none"),
]


@pytest.mark.parametrize("text,pick,place", CASES)
def test_pick_place(text, pick, place):
    info = language_only_task_info(text)
    assert info["pick"] == pick
    assert info["place"] == place
    assert pick in info["prompts"]


def test_no_bddl_fields():
    info = language_only_task_info("pick up the ketchup and place it in the basket")
    assert info["picks"] == ["ketchup"] and info["places"] == ["basket"] and info["pick_counts"] is None
