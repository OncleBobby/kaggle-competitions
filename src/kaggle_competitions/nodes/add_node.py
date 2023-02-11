from kedro.pipeline import *
from kedro.io import *
from kedro.runner import *

def add(x, y):
    return x + y

adder_node = node(func=add, inputs=["a", "b"], outputs="sum", name="adding_a_and_b")
print(str(adder_node))

print(adder_node.run(dict(a=2, b=3)))