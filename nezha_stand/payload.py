"""Pure-Python composite mass-property helpers for payload randomization."""


def parallel_axis_matrix(mass, offset):
    """Return m * (||r||^2 I - r r^T) for a three-vector offset."""
    x, y, z = (float(value) for value in offset)
    mass = float(mass)
    return [
        [mass * (y * y + z * z), -mass * x * y, -mass * x * z],
        [-mass * x * y, mass * (x * x + z * z), -mass * y * z],
        [-mass * x * z, -mass * y * z, mass * (x * x + y * y)],
    ]


def determinant3(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inverse_matrix3(matrix):
    """Invert a nonsingular 3x3 matrix."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = determinant3(matrix)
    if abs(determinant) <= 1.0e-12:
        raise ValueError("Inertia matrix is singular")
    scale = 1.0 / determinant
    return [
        [(e * i - f * h) * scale, (c * h - b * i) * scale,
         (b * f - c * e) * scale],
        [(f * g - d * i) * scale, (a * i - c * g) * scale,
         (c * d - a * f) * scale],
        [(d * h - e * g) * scale, (b * g - a * h) * scale,
         (a * e - b * d) * scale],
    ]


def is_positive_definite_symmetric(matrix):
    """Sylvester test for a symmetric 3x3 inertia matrix."""
    return (
        matrix[0][0] > 0.0
        and matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 > 0.0
        and determinant3(matrix) > 0.0
    )


def update_composite_mass_properties(old_mass, old_com, old_inertia,
                                     payload_delta_mass, payload_com,
                                     payload_inertia_per_kg):
    """Add a signed payload mass delta and return mass, COM and COM inertia.

    ``old_inertia`` is the full inertia matrix of the already-collapsed carrier,
    expressed about ``old_com``.  The payload's intrinsic inertia is diagonal
    and scales linearly with its mass.
    """
    old_mass = float(old_mass)
    delta_mass = float(payload_delta_mass)
    new_mass = old_mass + delta_mass
    if new_mass <= 0.0:
        raise ValueError("Payload randomization produced non-positive mass")

    old_com = [float(value) for value in old_com]
    payload_com = [float(value) for value in payload_com]
    new_com = [
        (old_mass * old_com[index] + delta_mass * payload_com[index]) / new_mass
        for index in range(3)
    ]
    old_offset = [old_com[index] - new_com[index] for index in range(3)]
    payload_offset = [
        payload_com[index] - new_com[index] for index in range(3)
    ]
    old_shift = parallel_axis_matrix(old_mass, old_offset)
    payload_shift = parallel_axis_matrix(delta_mass, payload_offset)

    new_inertia = []
    for row in range(3):
        values = []
        for column in range(3):
            intrinsic = (
                delta_mass * float(payload_inertia_per_kg[row])
                if row == column else 0.0
            )
            values.append(
                float(old_inertia[row][column])
                + old_shift[row][column]
                + intrinsic
                + payload_shift[row][column]
            )
        new_inertia.append(values)

    # Eliminate harmless float asymmetry before passing the tensor to PhysX.
    for row in range(3):
        for column in range(row + 1, 3):
            value = 0.5 * (
                new_inertia[row][column] + new_inertia[column][row]
            )
            new_inertia[row][column] = value
            new_inertia[column][row] = value
    if not is_positive_definite_symmetric(new_inertia):
        raise ValueError("Payload randomization produced invalid inertia")
    return new_mass, new_com, new_inertia
