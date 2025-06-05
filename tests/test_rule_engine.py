from rule_engine import (
    ColorReplacementRule,
    DiagonalFlipRule,
    RuleChain,
    RuleEngine,
    register_primary,
    PRIMARY_RULES,
)
from ingest import Example, Task


def test_color_replacement():
    rule = ColorReplacementRule(1, 3)
    grid = [[1, 2], [3, 1]]
    assert rule.apply(grid) == [[3, 2], [3, 3]]


def test_register_primary():
    rule = ColorReplacementRule(0, 1)
    register_primary("zero_to_one", rule)
    assert "zero_to_one" in PRIMARY_RULES

def test_rule_chain():
    flip = DiagonalFlipRule()
    replace = ColorReplacementRule(1, 9)
    chain = RuleChain(flip, replace)
    grid = [[1, 2], [3, 1]]
    assert chain.apply(grid) == [[9, 3], [2, 9]]


def test_learn_rule():
    engine = RuleEngine()
    engine.register("rep", ColorReplacementRule(1, 2))
    examples = [Example(index=0, input_grid=[[1]], output_grid=[[2]])]
    rule = engine.learn_rule(examples)
    assert isinstance(rule, ColorReplacementRule)


def test_solve_task_with_chain():
    engine = RuleEngine()
    engine.register("flip", DiagonalFlipRule())
    engine.register("rep", ColorReplacementRule(1, 9))
    train = [Example(index=0, input_grid=[[1, 2], [3, 1]], output_grid=[[9, 3], [2, 9]])]
    tests = [Example(index=0, input_grid=[[1, 4], [5, 1]], output_grid=[[9, 5], [4, 9]])]
    task = Task(id="t", metadata_xml="", training=train, tests=tests)
    preds = engine.solve_task(task, max_chain_length=2)
    assert preds == [[[9, 5], [4, 9]]]

