"""wcl.dissection must reproduce the code behind the dissertation's 2D results.

Reference: reproduce/dissection/experiments/lattice2d.py (label decoding, isometries,
matching) and exp358_zhou_duncan_common_boundary_optimizer.py (BACO). Test data: the
six-piece cat -> fish lattice labels of the dissertation (tests/data).
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reproduce" / "dissection"
sys.path[:0] = [str(ROOT), str(REF / "experiments"), str(REF)]
DATA = Path(__file__).resolve().parent / "data"

from wcl.dissection import baco, lattice                     # noqa: E402
import lattice2d as ref_lattice                               # noqa: E402
import exp358_zhou_duncan_common_boundary_optimizer as ref    # noqa: E402

ARGS = types.SimpleNamespace(seed=2012, samples_per_piece=96, max_relation_samples=220, a_init_weight=0.65,
                             b_init_weight=0.35, a_contact_weight=8.0, b_contact_weight=14.0, a_exterior_weight=3.5,
                             b_exterior_weight=1.4, shape_weight=0.9, a_shape_weight=0.8, b_shape_weight=0.35,
                             fair_weight=8.0, edge_weight=0.8)


class TestDissection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        la, lb_raw, _ = ref_lattice.decode_label_pngs(DATA / "cat_fish_k6_A.png", DATA / "cat_fish_k6_B.png")
        cls.la, cls.lb_raw = la, lb_raw
        cls.lb_ref, cls.isos_ref, _ = ref_lattice.geometry_match_labels(la, lb_raw)
        cls.lb, cls.isos = lattice.match_labels(la, lb_raw)
        cls.labels = sorted((set(np.unique(la).tolist()) & set(np.unique(cls.lb).tolist()) & set(cls.isos)) - {0})

    def test_read_labels(self):
        la, lb = lattice.read_labels(DATA / "cat_fish_k6_A.png", DATA / "cat_fish_k6_B.png")
        self.assertTrue(np.array_equal(la, self.la) and np.array_equal(lb, self.lb_raw))

    def test_matching(self):
        self.assertTrue(np.array_equal(self.lb, self.lb_ref))
        for l, iso in self.isos.items():
            r = self.isos_ref[l]
            self.assertEqual((iso.rot, iso.flip, iso.dy, iso.dx, iso.mismatch), (r.rot, r.flip, r.dy, r.dx, r.mismatch))

    def test_isometry_roundtrip(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 200, (50, 2))
        for iso in self.isos.values():
            self.assertTrue(np.allclose(iso.inverse_points(iso.forward_points(p)), p))
            r = self.isos_ref[iso.label]
            self.assertTrue(np.array_equal(iso.forward_points(p), ref.forward_points(r, p)))

    def test_baco_residuals_and_solution(self):
        prob = baco.problem(self.la, self.lb, self.isos, self.labels)
        model = ref.build_model(self.la, self.lb_ref, self.isos_ref, self.labels, ARGS)
        x0 = prob.pack(prob.init)
        self.assertTrue(np.array_equal(x0, ref.pack(model, model.init_loops)))
        self.assertTrue(np.array_equal(baco.residuals(x0, prob), ref.residuals(x0, model, ARGS)))
        x = baco.solve(prob, max_nfev=3)
        from scipy.optimize import least_squares
        xr = least_squares(lambda z: ref.residuals(z, model, ARGS), x0, max_nfev=3, loss="soft_l1", f_scale=2.0, x_scale="jac").x
        self.assertTrue(np.array_equal(x, xr))
        a, b = baco.contact_rms(x, prob)
        cr = ref.contact_rms(xr, model)
        self.assertAlmostEqual(a, cr["a_contact_rms_px"], places=12)
        self.assertAlmostEqual(b, cr["b_contact_rms_px"], places=12)


if __name__ == "__main__":
    unittest.main()

