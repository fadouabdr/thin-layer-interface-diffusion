
import numpy as np
from dolfin import *

# Cell tags
COATING_TAG = 1
BULK_TAG = 2

# Facet tags
INNER_TAG = 10       # x = -h, core/coating boundary
OUTER_TAG = 20       # x = 1, exterior Dirichlet boundary
NEUMANN_TAG = 30     # y = 0 or y = 1
INTERFACE_TAG = 40   # x = 0, coating/bulk internal interface


def create_fitted_full_mesh(h, n_layer=8, n_bulk=64, ny=64):
    """
    Create a fitted triangular mesh for the full thin-layer geometry:

        coating: [-h,0] x [0,1]
        bulk:    [0,1] x [0,1]

    The grid is fitted to x=0 so that the coating/bulk interface is exactly
    represented by mesh facets.
    """

    x_left = np.linspace(-h, 0.0, n_layer + 1)
    x_right = np.linspace(0.0, 1.0, n_bulk + 1)[1:]
    xs = np.concatenate([x_left, x_right])
    ys = np.linspace(0.0, 1.0, ny + 1)

    nx_total = len(xs) - 1

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)

    num_vertices = len(xs) * len(ys)
    num_cells = 2 * nx_total * ny

    editor.init_vertices(num_vertices)
    editor.init_cells(num_cells)

    def vid(i, j):
        return j * len(xs) + i

    vertex_id = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            editor.add_vertex(vertex_id, Point(float(x), float(y)))
            vertex_id += 1

    cell_id = 0
    for j in range(ny):
        for i in range(nx_total):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)

            # Two triangles per rectangle
            editor.add_cell(cell_id, np.array([v00, v10, v11], dtype=np.uintp))
            cell_id += 1
            editor.add_cell(cell_id, np.array([v00, v11, v01], dtype=np.uintp))
            cell_id += 1

    editor.close()
    mesh.init()

    # Mark cells
    cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in cells(mesh):
        xmid = cell.midpoint().x()
        if xmid < 0.0:
            cell_markers[cell] = COATING_TAG
        else:
            cell_markers[cell] = BULK_TAG

    # Mark facets
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    facet_markers.set_all(0)

    tol = 1.0e-12

    for facet in facets(mesh):
        mp = facet.midpoint()
        x = mp.x()
        y = mp.y()

        if near(x, -h, tol):
            facet_markers[facet] = INNER_TAG
        elif near(x, 1.0, tol):
            facet_markers[facet] = OUTER_TAG
        elif near(y, 0.0, tol) or near(y, 1.0, tol):
            facet_markers[facet] = NEUMANN_TAG
        elif near(x, 0.0, tol):
            # This is the internal coating/bulk interface.
            facet_markers[facet] = INTERFACE_TAG

    return mesh, cell_markers, facet_markers
