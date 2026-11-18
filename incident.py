import unittest
from math import gcd

from archive.catalog import ArtifactKind
from archive.lookup import read
from calibration.admission import calibrate
from calibration.snapshot import decode_calibration
from evidence.batches import Batch
from evidence.revisions import record
from review.projection import ResidueClass
from review.reports import assess
from tests.fixtures import BATCH, XY, StoreTestCase, equation
from tests.scenario_check import ShiftResult
from tests.test_calibration import CalibrationTests
from tests.test_handoff import HandoffTests


class IncidentTests(StoreTestCase):
    def fixed_relation(self, calibration_id):
        artifact = read(self.db, ArtifactKind.CALIBRATION, calibration_id)
        return decode_calibration(artifact.data).relation

    def test_even_sum_survives_publication(self):
        _, calibration_id = self.parity()
        relation = self.fixed_relation(calibration_id)
        self.assertEqual(sum(relation.point) % 2, 0)
        self.assertTrue(all(sum(direction) % 2 == 0 for direction in relation.directions))

    def test_affine_plane_keeps_odd_pairs(self):
        batch = Batch("line-three", "tick", ("x", "y", "z"))
        ref = record(self.db, batch, "bench", "1", [equation({"x": 2, "y": 1, "z": 1}, 0)])
        calibration_id = calibrate(self.db, batch.batch_id, [ref])
        result = assess(self.db, calibration_id, [{"y": 1}, {"z": 1}], [1, 1])
        self.assertEqual(result.reachable, (ResidueClass(0, 1),) * 2)
        self.assertIs(result.jointly_reachable, True)

    def test_report_keeps_common_offsets(self):
        ref = self.marks([equation({"x": 1, "y": 1}, 0)])
        calibration_id = calibrate(self.db, BATCH.batch_id, [ref])
        relation = self.fixed_relation(calibration_id)
        self.assertEqual(sum(relation.point), 0)
        self.assertTrue(all(sum(direction) == 0 for direction in relation.directions))
        self.assertEqual(gcd(*(direction[0] for direction in relation.directions)), 1)
        impossible = assess(self.db, calibration_id, XY, [0, 1])
        possible = assess(self.db, calibration_id, XY, [1, -1])
        self.assertEqual(impossible.reachable, (ResidueClass(0, 1),) * 2)
        self.assertEqual(possible.reachable, impossible.reachable)
        self.assertEqual((impossible.jointly_reachable, possible.jointly_reachable), (False, True))


def main():
    suite = unittest.TestSuite([
        CalibrationTests("test_integer_inconsistent"),
        IncidentTests("test_even_sum_survives_publication"),
        IncidentTests("test_affine_plane_keeps_odd_pairs"),
        IncidentTests("test_report_keeps_common_offsets"),
        HandoffTests("test_revision_isolation"),
    ])
    runner = unittest.TextTestRunner(verbosity=2, resultclass=ShiftResult)
    return 0 if runner.run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
