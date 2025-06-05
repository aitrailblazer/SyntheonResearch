from rule_engine import ColorReplacementRule, register_primary, PRIMARY_RULES


def test_color_replacement():
    rule = ColorReplacementRule(1, 3)
    grid = [[1, 2], [3, 1]]
    assert rule.apply(grid) == [[3, 2], [3, 3]]


def test_register_primary():
    rule = ColorReplacementRule(0, 1)
    register_primary("zero_to_one", rule)
    assert "zero_to_one" in PRIMARY_RULES
