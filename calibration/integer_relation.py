from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerRelation:
    point: tuple[int, ...]
    directions: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        width = len(self.point)
        if not 1 <= width <= 8 or len(self.directions) > width:
            raise ValueError("invalid integer relation dimensions")
        for vector in (self.point, *self.directions):
            if len(vector) != width or any(type(value) is not int for value in vector):
                raise ValueError("an integer relation requires exact integer vectors")


def full_integer_relation(width):
    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError("invalid coordinate count")
    directions = tuple(tuple(int(column == axis) for column in range(width)) for axis in range(width))
    return IntegerRelation((0,) * width, directions)


def _dot(row, vector):
    return sum(a * b for a, b in zip(row, vector, strict=True))


def evaluate_linear_form(relation, row):
    return _dot(row, relation.point), tuple(_dot(row, direction) for direction in relation.directions)


def scale_vector(vector, factor):
    return tuple(factor * value for value in vector)


def translate_point(point, direction, multiple):
    return tuple(a + multiple * b for a, b in zip(point, direction, strict=True))


def apply_parameters(relation, particular, kernel):
    """Re-express the relation after restricting the parameter lattice.

    ``particular`` and every vector in ``kernel`` live in the integer
    parameter space of ``relation.directions``.  The result is the image of
    ``particular + Z-span(kernel)``, keeping every feasible offset.
    """
    if len(particular) != len(relation.directions):
        raise ValueError('parameter vector does not match the relation')
    if any(len(vector) != len(particular) for vector in kernel):
        raise ValueError('kernel vectors must share the parameter dimension')
    point = relation.point
    for parameter, direction in zip(particular, relation.directions, strict=True):
        point = translate_point(point, direction, parameter)
    directions = []
    for vector in kernel:
        image = (0,) * len(relation.point)
        for parameter, direction in zip(vector, relation.directions, strict=True):
            image = translate_point(image, direction, parameter)
        directions.append(image)
    return IntegerRelation(point, tuple(directions))
