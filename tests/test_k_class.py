import pytest

from anchor_regression.linear_model import KClassMixin


@pytest.mark.parametrize(
    "kappa, expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        ("fuller(1)", 1),
        ("fuller(0.2)", 2),
        ("FULLER(4)", 3),
        ("fuller", 1),
        ("FulLeR", 1),
        ("liml", 0),
        ("LIML", 0),
    ],
)
def test__fuller_alpha(kappa, expected):
    assert KClassMixin._fuller_alpha(kappa) == expected
