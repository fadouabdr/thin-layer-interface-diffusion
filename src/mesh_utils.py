from dolfin import MeshFunction, SubDomain, UnitSquareMesh, near


ROBIN_TAG = 1
DIRICHLET_TAG = 2
NEUMANN_TAG = 3


class RobinBoundary(SubDomain):
    """Robin boundary Gamma_R = {x_1 = 0}."""

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)


class DirichletBoundary(SubDomain):
    """Dirichlet boundary Gamma_D = {x_1 = 1}."""

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)


class NeumannBoundary(SubDomain):
    """Neumann boundaries Gamma_N = {x_2 = 0} union {x_2 = 1}."""

    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[1], 0.0) or near(x[1], 1.0)
        )


def create_square_mesh(resolution):
    """
    Create a uniformly triangulated unit square and mark its boundaries.

    Parameters
    ----------
    resolution : int
        Number of mesh subdivisions in each coordinate direction.

    Returns
    -------
    mesh
        Unit-square finite-element mesh.
    boundaries
        Facet markers for the Robin, Dirichlet, and Neumann boundaries.
    """
    mesh = UnitSquareMesh(resolution, resolution)

    facet_dim = mesh.topology().dim() - 1
    boundaries = MeshFunction("size_t", mesh, facet_dim, 0)

    RobinBoundary().mark(boundaries, ROBIN_TAG)
    DirichletBoundary().mark(boundaries, DIRICHLET_TAG)
    NeumannBoundary().mark(boundaries, NEUMANN_TAG)

    return mesh, boundaries