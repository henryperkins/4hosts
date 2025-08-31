import pytest

from services.classification_engine import HostParadigm
from models.paradigms import normalize_to_enum, normalize_to_internal_code, enum_value


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("revolutionary", HostParadigm.DOLORES),
        ("dolores", HostParadigm.DOLORES),
        (HostParadigm.DOLORES, HostParadigm.DOLORES),
        ("devotion", HostParadigm.TEDDY),
        ("teddy", HostParadigm.TEDDY),
        ("analytical", HostParadigm.BERNARD),
        ("bernard", HostParadigm.BERNARD),
        ("strategic", HostParadigm.MAEVE),
        ("maeve", HostParadigm.MAEVE),
    ],
)
def test_normalize_to_enum(inp, expected):
    assert normalize_to_enum(inp) == expected


@pytest.mark.parametrize(
    "inp, expected",
    [
        (HostParadigm.DOLORES, "dolores"),
        ("revolutionary", "dolores"),
        ("dolores", "dolores"),
        (HostParadigm.BERNARD, "bernard"),
        ("analytical", "bernard"),
        ("bernard", "bernard"),
        (HostParadigm.MAEVE, "maeve"),
        ("strategic", "maeve"),
        (HostParadigm.TEDDY, "teddy"),
        ("devotion", "teddy"),
    ],
)
def test_normalize_to_internal_code(inp, expected):
    assert normalize_to_internal_code(inp) == expected


def test_enum_value_round_trip():
    for hp in HostParadigm:
        assert enum_value(hp) == hp.value
        # ensure value -> enum -> internal code is stable
        assert normalize_to_internal_code(hp.value) in {"dolores", "teddy", "bernard", "maeve"}

