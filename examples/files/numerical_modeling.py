```python
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

def solve_heat_conduction_fem(nodes, elements, k, Q, dirichlet_bcs, neumann_bcs):
    """
    Solves the 2D steady-state heat conduction equation using FEM.

    Args:
        nodes: NumPy array of node coordinates (n_nodes x 2).
        elements: NumPy array of element connectivity (n_elements x 3 for triangles).
        k: Thermal conductivity (scalar).
        Q: Volumetric heat generation rate (scalar).
        dirichlet_bcs: Dictionary of Dirichlet boundary conditions. Keys are node indices, values are temperatures.
        neumann_bcs: Dictionary of Neumann boundary conditions. Keys are edge tuples (node1, node2), values are heat fluxes.

    Returns:
        NumPy array of nodal temperatures.
    """

    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    # Initialize global stiffness matrix and load vector (sparse for efficiency)
    K = sparse.lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Loop over elements
    for element in elements:
        node_ids = element
        coords = nodes[node_ids]

        # Calculate element area (using Heron's formula)
        a = np.linalg.norm(coords[1] - coords[0])
        b = np.linalg.norm(coords[2] - coords[1])
        c = np.linalg.norm(coords[0] - coords[2])
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # Calculate element stiffness matrix (linear triangular element)
        b_coeffs = np.array([coords[1, 1] - coords[2, 1], coords[2, 1] - coords[0, 1], coords[0, 1] - coords[1, 1]])
        c_coeffs = np.array([coords[2, 0] - coords[1, 0], coords[0, 0] - coords[2, 0], coords[1, 0] - coords[0, 0]])

        Ke = (k / (4 * area)) * np.outer(b_coeffs, b_coeffs) + (k / (4 * area)) * np.outer(c_coeffs, c_coeffs)

        # Assemble into global stiffness matrix
        for i in range(3):
            for j in range(3):
                K[node_ids[i], node_ids[j]] += Ke[i, j]

        # Assemble load vector (assuming constant heat generation)
        Fe = (Q * area / 3) * np.ones(3)
        for i in range(3):
            F[node_ids[i]] += Fe[i]

    # Apply Neumann boundary conditions
    for edge, flux in neumann_bcs.items():
        node1, node2 = edge
        edge_length = np.linalg.norm(nodes[node1] - nodes[node2])
        F[node1] += flux * edge_length / 2
        F[node2] += flux * edge_length / 2

    # Apply Dirichlet boundary conditions
    dirichlet_nodes = list(dirichlet_bcs.keys())
    known_temperatures = np.array(list(dirichlet_bcs.values()))

    # Modify K and F to enforce Dirichlet BCs (using the penalty method or direct substitution)
    # Using direct substitution here for simplicity. This is less robust for large problems.
    for node, temperature in dirichlet_bcs.items():
        K[node, :] = 0  # Zero out the row
        K[node, node] = 1  # Set diagonal to 1
        F[node] = temperature  # Set the corresponding value in F

    # Solve the system of equations
    T = splinalg.cg(K.tocsr(), F)[0] # Using Conjugate Gradient method

    return T


if __name__ == '__main__':
    # Example Usage:  Simple Square Plate

    # Define nodes (example: square plate with 9 nodes)
    nodes = np.array([
        [0, 0],  # 0
        [1, 0],  # 1
        [2, 0],  # 2
        [0, 1],  # 3
        [1, 1],  # 4
        [2, 1],  # 5
        [0, 2],  # 6
        [1, 2],  # 7
        [2, 2]   # 8
    ])

    # Define elements (example: 8 triangular elements)
    elements = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [3, 4, 7],
        [3, 7, 6],
        [4, 5, 8],
        [4, 8, 7]
    ])

    # Material properties and heat generation
    k = 1.0  # Thermal conductivity
    Q = 0.0  # Volumetric heat generation

    # Boundary conditions (Dirichlet and Neumann)
    dirichlet_bcs = {
        0: 100.0,  # Temperature at node 0
        2: 50.0,   # Temperature at node 2
        6: 100.0,  # Temperature at node 6
        8: 50.0    # Temperature at node 8
    }

    neumann_bcs = {
        (1,2): 0.0, # zero flux
        (0,1): 0.0
    }

    # Solve the heat conduction problem
    temperatures = solve_heat_conduction_fem(nodes, elements, k, Q, dirichlet_bcs, neumann_bcs)

    print("Nodal Temperatures:\n", temperatures)

    # Visualization (requires matplotlib)
    plt.tricontourf(nodes[:, 0], nodes[:, 1], elements, temperatures)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Temperature Distribution")
    plt.colorbar()
    plt.show()
```