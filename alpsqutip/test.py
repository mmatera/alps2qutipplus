import matplotlib.pyplot as plt

from alpsmodels import list_operators_in_alps_xml, model_from_alps_xml
from geometry import graph_from_alps_xml, list_graph_in_alps_xml
from model import SystemDescriptor
from utils import eval_expr

lattice_lib_file = "lib/lattices.xml"
model_lib_file = "lib/models.xml"


def test_eval_expr():
    parms = {"a": "J", "J": 2, "subexpr": "a*J"}
    test_cases = [
        ("2+a", 4),
        ("sqrt(2+a)", 2),
        ("0*rand()", 0),
        ("2*J", 4),
        ("sqrt(subexpr)", 2),
    ]
    for expr, expect in test_cases:
        result = eval_expr(expr, parms)
        assert expect == result, (
            f"evaluating {expr}"
            f"{expect} of type {type(expect)}"
            f"!= {result} of {type(result)}"
        )


def test_load():
    print("geometries:")
    for name in list_graph_in_alps_xml(lattice_lib_file):
        g = graph_from_alps_xml(
            lattice_lib_file, name, parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1}
        )
        print(g)
        fig = plt.figure()
        if g.lattice and g.lattice["dimension"] > 2:
            ax = fig.add_subplot(projection="3d")
            ax.set_proj_type("persp")
        else:
            ax = fig.add_subplot()
        ax.set_title(name)
        g.draw(ax)
        plt.savefig(f"lib/figs/{name}.png")
    print("models:")

    for modelname in list_operators_in_alps_xml(model_lib_file):
        print("\n       ", modelname)
        print(40 * "*")
        model = model_from_alps_xml(
            model_lib_file, modelname, parms={"Nmax": 3, "local_S": 0.5}
        )
        print(
            "site types:", {name: lb["name"] for name, lb in model.site_basis.items()}
        )


def test_all():
    models = list_operators_in_alps_xml(model_lib_file)
    graphs = list_graph_in_alps_xml(lattice_lib_file)

    for model_name in models:
        print(model_name, "\n", 10 * "*")
        for graph_name in graphs:
            g = graph_from_alps_xml(
                lattice_lib_file,
                graph_name,
                parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1},
            )
            model = model_from_alps_xml(
                model_lib_file,
                model_name,
                parms={"L": 3, "W": 3, "a": 1, "b": 1, "c": 1, "Nmax": 5},
            )
            try:
                system = SystemDescriptor(g, model, {})
                print(system.global_operators.keys())
            except Exception as e:
                print("   ", graph_name, "  [failed]", e)
                continue
            print("   ", graph_name, "  [OK]")
        print("\n-------------")


test_load()
test_all()
test_eval_expr()
