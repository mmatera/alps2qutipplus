import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
from numpy.random import rand

from alpsqutip.utils import eval_expr, find_ref, next_name


def list_graph_in_alps_xml(filename="lattices.xml"):
    result = []
    xmltree = ET.parse(filename)
    lattices = xmltree.getroot()

    for g in lattices.findall("./GRAPH"):
        name = g.attrib.get("name", None)
        if name:
            result.append(name)

    # Otherwise, try with a lattice
    for lat in lattices.findall("./LATTICEGRAPH"):
        name = lat.attrib.get("name", None)
        if name:
            result.append(name)
    return tuple(result)


def graph_from_alps_xml(
    filename="lattices.xml", name="rectangular lattice", parms=None
):
    """
    Load from `filename` xml library a Graph or LatticeGraph of name
    `name`, using `parms` as parameters.
    """
    xmltree = ET.parse(filename)
    lattices = xmltree.getroot()
    if parms is None:
        parms = {}

    def process_coordinates(text: Optional[str] = None) -> Optional[list]:
        if text:
            return [eval_expr(c.strip(), parms) for c in text.split(" ")]
        return None

    def process_graph(node, parms):
        """Process a <GRAPH> node"""
        g_items = node.attrib
        parms = process_parms(node, parms)
        num_vertices = int(g_items["vertices"])
        vertices = {}
        edges = {}
        default_vertex = {"type": "0", "coords": None}

        for v in node.findall("./VERTEX"):
            v_items = process_vertex(v, parms)
            v_name = v_items.pop("id", None) or next_name(vertices)
            vertices[v_name] = v_items

        while len(vertices) < num_vertices:
            vertices[next_name(vertices)] = default_vertex.copy()

        for e in node.findall("./EDGE"):
            e_items = e.attrib
            list_edges = edges.get(e_items["type"], [])
            list_edges.append((e_items["source"], e_items["target"]))
            edges[e_items["type"]] = list_edges

        return GraphDescriptor(name=name, nodes=vertices, edges=edges, parms=parms)

    def process_lattice(node, parms):
        """Process a <LATTICE> node"""
        parms = process_parms(node, parms)
        lattice = {"basis": [], "reciprocal_basis": []}
        lattice["dimension"] = int(node.attrib.get("dimension", 0))
        lattice["name"] = node.attrib.get("name", "")
        basis = lattice["basis"]
        reciprocal_basis = lattice["reciprocal_basis"]

        for b in node.findall("./BASIS"):
            for v in b.findall("./VECTOR"):
                coords = process_coordinates(v.text)
                basis.append(coords)

        for b in node.findall("./RECIPROCALBASIS"):
            for v in b.findall("./VECTOR"):
                coords = [eval_expr(c.strip(), parms) for c in v.text.split(" ")]
                reciprocal_basis.append(coords)

        return lattice

    def process_latticegraph(node, parms):
        """Process a <LATTICEGRAPH> node"""
        parms = process_parms(node, parms)
        unitcell = []
        vertices = {}
        edges = {}

        for uc in node.findall("./UNITCELL"):
            uc = find_ref(uc, lattices)
            unitcell = process_unitcell(uc, parms)

        for fl in node.findall("./FINITELATTICE"):
            parms = process_parms(fl, parms)
            dimensions = unitcell["dimension"]
            extents = []
            if dimensions:
                bcs = dimensions * ["open"]
                for bc in fl.findall("./BOUNDARY"):
                    bc_items = bc.attrib
                    if "dimension" in bc_items:
                        c_dim = int(bc_items.get("dimension", 1)) - 1
                        bcs[c_dim] = bc_items.get("type", bcs[c_dim])
                    else:
                        bcs = dimensions * bc_items.get("type", bcs[0])

                extents = dimensions * [0]
                for ext in fl.findall("./EXTENT"):
                    ext_items = ext.attrib
                    c_dim = int(ext_items.get("dimension", 1)) - 1
                    size = ext_items.get("size", extents[c_dim])
                    while size in parms:
                        size = parms[size]
                    extents[c_dim] = int(size)

            for lat_entry in fl.findall("./LATTICE"):
                lattice = process_lattice(find_ref(lat_entry, lattices), parms)

        # Build a list of cells
        dim = dimensions
        cells = []
        curr_coords = (dim + 1) * [0]
        while True:
            cells.append(tuple(curr_coords[:-1]))
            d = 0
            curr_coords[d] += 1
            while d < dim and curr_coords[d] == extents[d]:
                curr_coords[d] = 0
                d += 1
                curr_coords[d] += 1

            if curr_coords[-1]:
                break

        # Add the vertices associated to each cell
        cell_vertices = unitcell["vertices"]
        lattice_basis = [np.array(v) for v in lattice["basis"]]
        for cell in cells:
            for vertex, v_attr in cell_vertices.items():
                coords = v_attr.get("coords", None)
                if coords is not None:
                    v_attr = v_attr.copy()
                    v_attr["coords"] = (
                        sum(c * b for c, b in zip(cell, lattice_basis)) + coords
                    )

                vertices[f"{vertex}{list(cell)}"] = v_attr

        # Add inhomogeneous vertices
        for inhomogeneous in node.findall("./INHOMOGENEOUS"):
            for v in inhomogeneous.findall("VERTEX"):
                v_items = process_vertex(v, parms)
                v_name = v_items.get("name", None) or next_name(vertices, 1, "defect_")
                vertices[v_name] = v_items

        # Build edges
        cell_edges = unitcell["edges"]
        for et, bond_desc_list in cell_edges.items():
            bonds = []
            for cell in cells:
                for bd in bond_desc_list:
                    src_name = bd["src"]
                    src = list(u + d for u, d in zip(cell, bd["offset_src"]))
                    tgt_name = bd["tgt"]
                    tgt = list(u + d for u, d in zip(cell, bd["offset_tgt"]))
                    skip = False
                    for d in range(dim):
                        if bcs[d] == "periodic":
                            if src[d] >= extents[d] or src[d] < 0:
                                src[d] = src[d] % extents[d]
                            if tgt[d] >= extents[d] or tgt[d] < 0:
                                tgt[d] = tgt[d] % extents[d]
                        else:
                            if (
                                src[d] >= extents[d]
                                or tgt[d] >= extents[d]
                                or src[d] < 0
                                or tgt[d] < 0
                            ):
                                skip = True
                    if skip:
                        continue
                    new_bond = (
                        f"{src_name}{src}",
                        f"{tgt_name}{tgt}",
                    )
                    bonds.append(new_bond)
            edges[et] = bonds

        return GraphDescriptor(
            name=name, nodes=vertices, edges=edges, lattice=lattice, parms=parms
        )

    def process_parms(node, parms):
        """Process tje <PARAMETER>s nodes"""
        default_parms = {}
        for parameter in node.findall("./PARAMETER"):
            key_vals = parameter.attrib
            default_parms[key_vals["name"]] = key_vals["default"]
        default_parms.update(parms)
        return default_parms

    def process_unitcell(node, parms):
        parms = process_parms(node, parms)
        vertices = {}
        edges = {}
        unitcell = {"vertices": vertices, "edges": edges}
        dimension = int(node.attrib.get("dimension", 0))
        unitcell["dimension"] = dimension
        unitcell["name"] = node.attrib.get("name", "")

        for v in node.findall("./VERTEX"):
            v_items = process_vertex(v, parms)
            name = v_items.pop("name", None) or next_name(vertices)
            vertices[name] = v_items

        for e in node.findall("./EDGE"):
            e_items = e.attrib
            e_type = e_items.get("type", "0")
            e_src = e_items.get("source", "0")
            e_tgt = e_items.get("target", "0")
            e_offset_src = ""
            e_offset_tgt = ""
            for src in e.findall("./SOURCE"):
                src_items = src.attrib
                e_src = src_items.get("vertex", e_src)
                e_offset_src = src_items.get("offset", e_offset_src)
            for tgt in e.findall("./TARGET"):
                tgt_items = tgt.attrib
                e_tgt = tgt_items.get("vertex", e_tgt)
                e_offset_tgt = tgt_items.get("offset", e_offset_tgt)

            e_offset_src = (
                e_offset_src.split(" ") if e_offset_src else dimension * ["0"]
            )
            e_offset_tgt = (
                e_offset_tgt.split(" ") if e_offset_tgt else dimension * ["0"]
            )
            e_offset_src = [int(c) for c in e_offset_src]
            e_offset_tgt = [int(c) for c in e_offset_tgt]

            edges_type_list = edges.get(e_type, [])
            edges_type_list.append(
                {
                    "src": e_src,
                    "tgt": e_tgt,
                    "offset_src": e_offset_src,
                    "offset_tgt": e_offset_tgt,
                }
            )
            edges[e_type] = edges_type_list

        return unitcell

    def process_vertex(node, parms):
        """Process a <VERTEX> node"""
        v_attributes = node.attrib
        v_attributes["type"] = v_attributes.get("type", "0")
        v_attributes["coords"] = None
        for c in node.findall("./COORDINATE"):
            v_attributes["coords"] = process_coordinates(c.text)
        return v_attributes

    # Try to find a graph
    for g in lattices.findall("./GRAPH"):
        if ("name", name) in g.items():
            return process_graph(g, parms)

    # Otherwise, try with a lattice
    for lat in lattices.findall("./LATTICEGRAPH"):
        if ("name", name) in lat.items():
            return process_latticegraph(lat, parms)

    return None


class GraphDescriptor:
    """
    A description of a Graph
    """

    name: str
    nodes: dict
    edges: dict
    parms: dict

    def __init__(
        self,
        name: str,
        nodes: dict,
        edges: Optional[dict],
        lattice: Optional[dict] = None,
        parms: Optional[dict] = None,
    ):
        self.name = name
        self.nodes = nodes
        self.edges = edges or {}
        self.lattice = lattice or None
        self.parms = parms or {}
        self.complete_coordiantes()

    def complete_coordiantes(self):
        nodes = self.nodes
        lattice = self.lattice
        # TODO: it would be great to use a better algorithm to
        # build the missing coordinates.
        if lattice is not None:
            basis = [np.array(b) for b in lattice["basis"]]
            dim = len(basis[0])
            local_coords = {}
            for name, n_attr in nodes.items():
                if n_attr.get("coords", None) is not None:
                    continue
                n_attr = n_attr.copy()
                nodes[name] = n_attr
                if "[" in name:
                    v_name, cell_str = name.split("[")
                    cell = [int(c) for c in cell_str[:-1].strip().split(",")]
                    l_coords = local_coords.get(v_name, rand(dim))
                    local_coords[v_name] = l_coords
                    n_attr["coords"] = (
                        sum(c * b for c, b in zip(cell, basis)) + l_coords
                    )
                else:
                    n_attr["coords"] = rand(dim)
        else:
            for name, n_attr in nodes.items():
                if n_attr.get("coords", None) is not None:
                    continue
                n_attr = n_attr.copy()
                nodes[name] = n_attr
                n_attr["coords"] = rand(2)

    def __repr__(self):
        result = (
            f"Graph {self.name}. Vertices: \n  "
            + "\n  ".join([f" {i} of type {t}" for i, t in self.nodes.items()])
            + "\nEdges"
        )

        for t, bnds in self.edges.items():
            result += f"\n type {t}:\n    " + "\n    ".join(
                f"{b[0]}-{b[1]}" for b in bnds
            )
        return result

    def draw(self, ax):
        coords = {}
        nodes = self.nodes
        edges = self.edges
        if self.lattice and self.lattice["dimension"] > 2:
            for name in self.nodes:
                coords[name] = nodes[name]["coords"][:3]

        elif self.lattice and self.lattice["dimension"] == 1:
            for name in self.nodes:
                coords[name] = np.array([nodes[name]["coords"][0], 0.0])
        else:
            for name in self.nodes:
                coords[name] = nodes[name]["coords"][:2]

        for name, x in coords.items():
            ax.scatter(*[[u] for u in x])

        for t, bl in edges.items():
            for src_name, tgt_name in bl:
                src, tgt = coords[src_name], coords[tgt_name]
                # TODO: color by type t
                ax.plot(*[[u, v] for u, v in zip(src, tgt)])
