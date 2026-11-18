from math import gcd

from calibration.integer_relation import (
    IntegerRelation, evaluate_linear_form, scale_vector, translate_point,
)


def apply_congruence(relation, row, rhs, modulus):
    if type(modulus) is not int or modulus <= 0:
        raise ValueError("modulus must be a positive integer")
    if modulus == 1:
        return relation
    base, coefficients = evaluate_linear_form(relation, row)
    delta = rhs - base
    if delta % gcd(modulus, *coefficients):
        return None

    point = relation.point
    directions = []
    for index, (coefficient, direction) in enumerate(zip(coefficients, relation.directions, strict=True)):
        others = coefficients[:index] + coefficients[index + 1:]
        projected_modulus = gcd(modulus, *others)
        common = gcd(coefficient, projected_modulus)
        period = projected_modulus // common
        if period > 1:
            inverse = pow(coefficient // common, -1, period)
            parameter = (delta // common * inverse) % period
            point = translate_point(point, direction, parameter)
        directions.append(scale_vector(direction, period))
    return IntegerRelation(point, tuple(directions))
