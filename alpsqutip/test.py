import matplotlib.pyplot as plt

from alpsmodels import list_operators_in_alps_xml, model_from_alps_xml
from geometry import graph_from_alps_xml, list_graph_in_alps_xml
from utils import eval_expr

lattice_lib_file = "lib/lattices.xml"
model_lib_file = "lib/models.xml"


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


test_load()
