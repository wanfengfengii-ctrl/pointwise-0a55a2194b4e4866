from calibration.integer_relation import apply_parameters, evaluate_linear_form
from calibration.lattices import solve_congruence


def apply_congruence(relation, row, rhs, modulus):
    if type(modulus) is not int or modulus <= 0:
        raise ValueError("modulus must be a positive integer")
    if modulus == 1:
        return relation
    base, coefficients = evaluate_linear_form(relation, row)
    solution = solve_congruence(coefficients, rhs - base, modulus)
    if solution is None:
        return None
    particular, kernel = solution
    return apply_parameters(relation, particular, kernel)
