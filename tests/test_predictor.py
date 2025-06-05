from ingest import Example, Task
from predictor import (
    learn_color_map_rule,
    suggest_color_map_rule,
    SymbolicPredictor,
)


def test_learn_color_map_rule():
    examples = [
        Example(index=0, input_grid=[[1, 2]], output_grid=[[3, 4]]),
        Example(index=1, input_grid=[[2, 1]], output_grid=[[4, 3]]),
    ]
    rule = learn_color_map_rule(examples)
    assert rule is not None
    assert rule.apply([[1, 2, 2]]) == [[3, 4, 4]]


def test_symbolic_predictor():
    training = [Example(index=0, input_grid=[[1]], output_grid=[[2]])]
    tests = [Example(index=0, input_grid=[[1]], output_grid=[[2]])]
    task = Task(id="t1", metadata_xml="", training=training, tests=tests)
    predictor = SymbolicPredictor()
    predictor.learn(task)
    preds = predictor.predict(task)
    assert preds == [[[2]]]


def test_suggest_color_map_rule():
    predicted = [[[1]]]
    expected = [[[2]]]
    rule = suggest_color_map_rule(predicted, expected)
    assert rule is not None
    assert rule.apply([[1]]) == [[2]]
