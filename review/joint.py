from calibration.integer_relation import apply_parameters, evaluate_linear_form
from calibration.lattices import solve_equation
from review.projection import ResidueClass, project_channel


def _restrict(relation, coefficients, value):
    base, dots = evaluate_linear_form(relation, coefficients)
    solution = solve_equation(dots, value - base)
    if solution is None:
        return None
    particular, kernel = solution
    return apply_parameters(relation, particular, kernel)


def evaluate_joint_claim(relation, channels):
    """Decide whether one common integer offset satisfies both claims.

    A false marginal makes the pair unreachable; otherwise both equations
    are imposed on the same fixed lattice, so the second equation sees the
    offsets still available after the first one was satisfied.
    """
    restricted = relation
    for channel in channels:
        reachable = project_channel(relation, channel.coefficients)
        if isinstance(reachable, ResidueClass):
            if channel.claimed % reachable.step != reachable.residue:
                return False
        elif channel.claimed != reachable:
            return False
        restricted = _restrict(restricted, channel.coefficients, channel.claimed)
        if restricted is None:
            return False
    return True
