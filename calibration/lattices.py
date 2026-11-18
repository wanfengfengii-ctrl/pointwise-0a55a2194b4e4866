"""Exact integer lattice arithmetic for one linear constraint at a time.

A fixed relation is an affine lattice: a particular integer point plus the
integer span of a set of independent integer directions.  Applying an
equation or a congruence must return exactly the sublattice of points that
also satisfies the new constraint, or ``None`` when no integer solution
exists.  Everything here works with unbounded Python integers.
"""

from math import gcd


def extended_gcd(left, right):
    """Return ``(g, s, t)`` with ``g >= 0`` and ``s * left + t * right == g``."""
    old_r, r = left, right
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def primitive_completion(vector):
    """Return a unimodular matrix whose first column expresses ``vector``.

    ``vector`` must be primitive (its entries have gcd one).  The returned
    matrix ``w`` is unimodular, ``sum_i vector[i] * w[i][0] == 1`` and every
    later column is an integer kernel vector of ``vector``.
    """
    values = [int(value) for value in vector]
    size = len(values)
    if not size:
        raise ValueError('cannot complete an empty vector')
    matrix = [[int(row == column) for column in range(size)] for row in range(size)]

    def column(index):
        return [matrix[row][index] for row in range(size)]

    def store(index, entries):
        for row in range(size):
            matrix[row][index] = entries[row]

    while True:
        nonzero = [index for index, value in enumerate(values) if value]
        if len(nonzero) <= 1:
            break
        first, second = nonzero[0], nonzero[1]
        divisor, left, right = extended_gcd(values[first], values[second])
        first_column, second_column = column(first), column(second)
        pivot = -(values[second] // divisor)
        partner = values[first] // divisor
        store(first, [left * a + right * b
                      for a, b in zip(first_column, second_column, strict=True)])
        store(second, [pivot * a + partner * b
                       for a, b in zip(first_column, second_column, strict=True)])
        values[first], values[second] = divisor, 0

    nonzero = [index for index, value in enumerate(values) if value]
    if not nonzero:
        raise ValueError('a zero vector cannot be completed')
    pivot = nonzero[0]
    if values[pivot] == -1:
        store(pivot, [-value for value in column(pivot)])
        values[pivot] = 1
    if values[pivot] != 1:
        raise ValueError('vector must be primitive')
    if pivot:
        first_column, pivot_column = column(0), column(pivot)
        store(0, pivot_column)
        store(pivot, first_column)
    return tuple(tuple(row) for row in matrix)


def _particular_from(matrix, factor, width):
    return tuple(factor * matrix[row][0] for row in range(width))


def solve_equation(coefficients, delta):
    """Solve ``sum(coefficients[i] * z[i]) == delta`` over the integers.

    Returns ``(particular, kernel_basis)`` or ``None`` when unsolvable.
    """
    width = len(coefficients)
    divisor = 0
    for coefficient in coefficients:
        divisor = gcd(divisor, coefficient)
    if divisor == 0:
        if delta != 0:
            return None
        return tuple(0 for _ in range(width)), tuple(
            tuple(int(row == column) for row in range(width)) for column in range(width)
        )
    if delta % divisor:
        return None
    primitive = tuple(coefficient // divisor for coefficient in coefficients)
    matrix = primitive_completion(primitive)
    factor = delta // divisor
    particular = _particular_from(matrix, factor, width)
    kernel = tuple(tuple(matrix[row][index] for row in range(width))
                   for index in range(1, width))
    return particular, kernel


def solve_congruence(coefficients, delta, modulus):
    """Solve ``sum(coefficients[i] * z[i]) == delta (mod modulus)``.

    Returns ``(particular, kernel_basis)`` or ``None`` when unsolvable.
    """
    width = len(coefficients)
    if width == 0:
        return ((), ()) if delta % modulus == 0 else None
    divisor = modulus
    for coefficient in coefficients:
        divisor = gcd(divisor, coefficient)
    if delta % divisor:
        return None
    # a.z - modulus.t == delta with one auxiliary integer t.
    augmented = tuple(coefficient // divisor for coefficient in coefficients)
    augmented += (-(modulus // divisor),)
    matrix = primitive_completion(augmented)
    factor = delta // divisor
    particular = _particular_from(matrix, factor, width)
    kernel = tuple(tuple(matrix[row][index] for row in range(width))
                   for index in range(1, width + 1))
    return particular, kernel
