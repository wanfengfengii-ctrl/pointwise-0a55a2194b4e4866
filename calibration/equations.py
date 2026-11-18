from fractions import Fraction
from math import gcd

from calibration.integer_relation import (
    IntegerRelation, evaluate_linear_form, scale_vector, translate_point,
)


def apply_equation(relation, row, rhs):
    base, coefficients = evaluate_linear_form(relation, row)
    delta = rhs - base
    pivot_index = next((index for index, value in enumerate(coefficients) if value), None)
    if pivot_index is None:
        return relation if delta == 0 else None

    pivot = coefficients[pivot_index]
    pivot_direction = relation.directions[pivot_index]
    point = translate_point(relation.point, pivot_direction, int(Fraction(delta, pivot)))
    directions = []
    for index, (coefficient, direction) in enumerate(zip(coefficients, relation.directions, strict=True)):
        if index == pivot_index:
            continue
        step = abs(pivot) // gcd(pivot, coefficient)
        compensation = -(coefficient * step) // pivot
        directions.append(translate_point(scale_vector(direction, step), pivot_direction, compensation))
    return IntegerRelation(point, tuple(directions))
