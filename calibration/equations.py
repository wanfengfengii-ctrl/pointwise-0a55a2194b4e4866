from calibration.integer_relation import apply_parameters, evaluate_linear_form
from calibration.lattices import solve_equation


def apply_equation(relation, row, rhs):
    base, coefficients = evaluate_linear_form(relation, row)
    solution = solve_equation(coefficients, rhs - base)
    if solution is None:
        return None
    particular, kernel = solution
    return apply_parameters(relation, particular, kernel)
