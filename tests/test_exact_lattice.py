"""Regressions for exact integer-lattice calibration and joint review.

The production code is cross-checked against direct enumeration of the
feasible integer offsets, so a truncating equation step, a wrong congruence
basis or a marginal-only joint test fails here even when the history-specific
assertions still hold.
"""

import itertools
import random
from fractions import Fraction

from archive.catalog import ArtifactKind, list_artifacts
from archive.lookup import read
from calibration.admission import InconsistentEvidence, admit_constraints, calibrate
from calibration.congruences import apply_congruence
from calibration.equations import apply_equation
from calibration.integer_relation import full_integer_relation
from calibration.snapshot import decode_calibration
from evidence.observations import Observation
from evidence.revisions import fetch_revision
from review.channels import ChannelClaim
from review.joint import evaluate_joint_claim
from review.projection import ResidueClass, project_channel
from tests.fixtures import BATCH, StoreTestCase, congruence, equation


def satisfies(coefficients, point, rhs, modulus):
    total = sum(coefficient * value for coefficient, value in zip(coefficients, point, strict=True))
    return total == rhs if modulus is None else (total - rhs) % modulus == 0


def brute_force(constraints, width, bound):
    return {
        point
        for point in itertools.product(range(-bound, bound + 1), repeat=width)
        if all(satisfies(coefficients, point, rhs, modulus)
               for coefficients, rhs, modulus in constraints)
    }


def nullspace_membership(relation, point):
    """Check membership by solving for rational parameters and testing integrality.

    Independent of the production solver: the directions are first verified
    independent with ordinary Gaussian elimination, after which the parameter
    vector is unique.
    """
    width = len(relation.point)
    target = [point[index] - relation.point[index] for index in range(width)]
    columns = len(relation.directions)
    matrix = [[Fraction(relation.directions[column][row]) for column in range(columns)]
              + [Fraction(target[row])] for row in range(width)]
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, width) if matrix[row][column]), None)
        if pivot is None:
            return False
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        lead = matrix[pivot_row][column]
        matrix[pivot_row] = [value / lead for value in matrix[pivot_row]]
        for row in range(width):
            if row != pivot_row and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [value - factor * other
                               for value, other in zip(matrix[row], matrix[pivot_row], strict=True)]
        pivot_row += 1
    for row in range(pivot_row, width):
        if matrix[row][columns] != 0:
            return False
    return all(matrix[row][columns].denominator == 1 for row in range(pivot_row))


def assert_sound(relation, constraints):
    """The representative satisfies every constraint; directions move within its kernel."""
    for coefficients, rhs, modulus in constraints:
        vectors = ((relation.point, rhs), *((direction, 0) for direction in relation.directions))
        for vector, target in vectors:
            total = sum(coefficient * value
                        for coefficient, value in zip(coefficients, vector, strict=True))
            if modulus is None:
                assert total == target, (coefficients, rhs, modulus, vector)
            else:
                assert (total - target) % modulus == 0, (coefficients, rhs, modulus, vector)


def images(relation, channel, bound):
    projected = project_channel(relation, channel)
    if isinstance(projected, ResidueClass):
        limit = bound * sum(abs(value) for value in channel) + abs(projected.residue) + projected.step
        return {
            value for value in range(-limit, limit + 1)
            if (value - projected.residue) % projected.step == 0
        }
    return {projected}


class ExactLatticeRegressionTests(StoreTestCase):
    def test_unsolvable_systems_are_rejected(self):
        for revision, observations in enumerate((
            [equation({'x': 1, 'y': 1}, 0), equation({'x': 1, 'y': -1}, 1)],
            [equation({'x': 2}, 1)],
            [congruence({'x': 4}, 1, 6)],
            [equation({'x': 2, 'y': 2}, 1)],
        )):
            with self.subTest(observations=observations):
                ref = self.marks(observations, f'reject-{revision}')
                with self.assertRaises(InconsistentEvidence):
                    calibrate(self.db, BATCH.batch_id, [ref])
                self.assertEqual(list_artifacts(self.db, ArtifactKind.CALIBRATION), ())
                self.assertEqual(fetch_revision(self.db, ref).observations[0].rhs,
                                 observations[0]['rhs'])

    def test_fixed_relation_is_exactly_the_feasible_set(self):
        constraints = [((1, 1), 1, None), ((1, 0), 3, 4)]
        ref = self.marks([equation({'x': 1, 'y': 1}, 1), congruence({'x': 1}, 3, 4)])
        calibration_id = calibrate(self.db, BATCH.batch_id, [ref])
        relation = decode_calibration(
            read(self.db, ArtifactKind.CALIBRATION, calibration_id).data).relation
        assert_sound(relation, constraints)
        feasible = brute_force(constraints, 2, 9)
        for point in feasible:
            self.assertTrue(nullspace_membership(relation, point), point)
        for channel in ((1, 0), (0, 1), (1, 1), (1, -1), (2, -3)):
            achieved = {sum(coefficient * value
                            for coefficient, value in zip(channel, point, strict=True))
                        for point in feasible}
            self.assertTrue(achieved <= images(relation, channel, 9), channel)

    def test_parity_relation_keeps_odd_sum_fluid(self):
        even = admit_constraints(2, (Observation((1, 1), 0, 2),))
        for point in brute_force([((1, 1), 0, 2)], 2, 5):
            self.assertTrue(nullspace_membership(even, point), point)
        self.assertEqual(len(even.directions), 2)
        self.assertFalse(evaluate_joint_claim(even, [
            ChannelClaim((1, 0), 0), ChannelClaim((0, 1), 1)]))
        self.assertTrue(evaluate_joint_claim(even, [
            ChannelClaim((1, 0), 1), ChannelClaim((0, 1), 1)]))
        unrestricted = admit_constraints(2, ())
        self.assertTrue(evaluate_joint_claim(unrestricted, [
            ChannelClaim((1, 0), 5), ChannelClaim((0, 1), -3)]))

    def test_exact_locks_both_channels(self):
        exact = admit_constraints(2, (Observation((1, 0), 2), Observation((0, 1), -1)))
        self.assertEqual(project_channel(exact, (1, 0)), 2)
        self.assertEqual(project_channel(exact, (0, 1)), -1)
        self.assertTrue(evaluate_joint_claim(exact, [
            ChannelClaim((1, 0), 2), ChannelClaim((0, 1), -1)]))
        self.assertFalse(evaluate_joint_claim(exact, [
            ChannelClaim((1, 0), 2), ChannelClaim((0, 1), 1)]))

    def test_seeded_brute_force_cross_check(self):
        generator = random.Random(20241118)
        for _ in range(80):
            width = generator.randint(1, 3)
            relation = full_integer_relation(width)
            constraints = []
            for _ in range(generator.randint(1, 3)):
                coefficients = tuple(generator.randint(-3, 3) for _ in range(width))
                if generator.random() < 0.5:
                    rhs = generator.randint(-4, 4)
                    relation = apply_equation(relation, coefficients, rhs)
                    constraints.append((coefficients, rhs, None))
                else:
                    modulus = generator.randint(2, 5)
                    rhs = generator.randint(-4, 4)
                    relation = apply_congruence(relation, coefficients, rhs, modulus)
                    constraints.append((coefficients, rhs, modulus))
                if relation is None:
                    break
            if relation is None:
                self.assertFalse(brute_force(constraints, width, 25), constraints)
                continue
            assert_sound(relation, constraints)
            feasible = brute_force(constraints, width, 6)
            for point in feasible:
                self.assertTrue(nullspace_membership(relation, point), (point, constraints))
            for channel in itertools.chain(
                (tuple(int(slot == axis) for slot in range(width)) for axis in range(width)),
                (tuple(generator.randint(-2, 2) for _ in range(width)) for _ in range(4)),
            ):
                if not any(channel):
                    continue
                achieved = {sum(coefficient * value
                                for coefficient, value in zip(channel, point, strict=True))
                            for point in feasible}
                self.assertTrue(achieved <= images(relation, channel, 6),
                                (channel, constraints))
            wide = brute_force(constraints, width, 25)
            if wide:
                witness = min(wide)
                channels = []
                for _ in range(2):
                    coefficients = tuple(generator.randint(-3, 3) for _ in range(width))
                    claimed = sum(coefficient * value for coefficient, value
                                  in zip(coefficients, witness, strict=True))
                    channels.append(ChannelClaim(coefficients, claimed))
                self.assertTrue(evaluate_joint_claim(relation, channels),
                                (channels, constraints, witness))
            for _ in range(2):
                channels = [ChannelClaim(
                    tuple(generator.randint(-3, 3) for _ in range(width)),
                    generator.randint(-3, 3),
                ) for _ in range(2)]
                if not evaluate_joint_claim(relation, channels):
                    self.assertFalse(
                        any(all(sum(coefficient * value
                                    for coefficient, value
                                    in zip(channel.coefficients, point, strict=True)) == channel.claimed
                                for channel in channels) for point in wide),
                        (channels, constraints))
