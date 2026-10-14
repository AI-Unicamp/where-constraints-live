"""wcl.csg must reproduce the code that produced the dissertation's Chapter 3 results.

The reference implementation (reproduce/inverse_csg/siggraph_csg_benchmark_v3.py and
csg_leaf_budget_ablation_stable_fixed.py) is imported directly and compared on random
inputs. Equality is exact (torch.equal) unless stated otherwise.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproduce" / "inverse_csg"))

import wcl.csg as csg                                              # noqa: E402
import siggraph_csg_benchmark_v3 as ref                            # noqa: E402
import csg_leaf_budget_ablation_stable_fixed as sweep              # noqa: E402


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


class TestPrimitives(unittest.TestCase):
    def test_sdfs_match_reference(self):
        p = rand(6, 50, 3, seed=1)
        raw = rand(6, 1, 3, seed=2)
        size = raw.abs() + ref.MIN_SIZE
        for t in range(5):
            self.assertTrue(torch.equal(csg.primitive_sdf(t, p, size), ref.sdf_by_type(t, p, raw)), csg.TYPES[t])

    def test_volume_radius_matches_reference(self):
        raw = rand(10, 3, seed=3)
        size = raw.abs() + ref.MIN_SIZE
        for t in range(5):
            self.assertTrue(torch.equal(csg.volume_radius(t, size), ref.volume_equiv_radius(t, raw)), csg.TYPES[t])

    def test_exact_volumes(self):
        # Monte-Carlo check of the exact wedge volumes stated in the docstrings
        g = torch.Generator().manual_seed(4)
        h = torch.tensor([0.7, 0.5, 0.9])
        x = (torch.rand(400_000, 3, generator=g) * 2 - 1) * h
        box = 8 * h.prod()
        for t, name in ((3, "wedge"), (4, "corner_wedge")):
            mc = box * (csg.primitive_sdf(t, x, h.expand_as(x)) <= 0).float().mean()
            self.assertAlmostEqual(float(mc), float(csg.primitive_volume(t, h)), delta=0.01 * float(box), msg=name)
        self.assertAlmostEqual(float(csg.primitive_volume(0, torch.tensor([2.0, 0, 0]))), 32 * math.pi / 3, places=4)


class TestSlots(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.slots = csg.TypedSlots(9, selector="sine_softmax")
        cfg = ref.PoolCfg(n_leaves=9, mode="sine_softmax", type_agg="product", volume_norm=True, omega=10.0)
        self.ref = ref.PrimitivePool(cfg)
        self.ref.load_state_dict(self.slots.state_dict())
        self.x = rand(64, 3, seed=8)

    def test_candidates(self):
        self.assertTrue(torch.equal(self.slots.candidate_occupancy(self.x), self.ref.occs_by_type(self.x)))

    def test_extractions(self):
        for beta in (1.0, 20.0):
            self.assertTrue(torch.equal(self.slots(self.x, beta, "soft"), self.ref.get_occs(self.x, beta, hard_leaf=False)))
            self.assertTrue(torch.equal(self.slots(self.x, beta, "onehot"), self.ref.get_occs(self.x, beta, hard_leaf=True)))
            stack = self.ref.occs_by_type(self.x)
            self.assertTrue(torch.equal(self.slots(self.x, beta, "multihot", 0.5),
                                        sweep.leaf_occs_from_stack(self.ref, stack, beta, "multihot", 0.5)))

    def test_multihot_selector(self):
        torch.manual_seed(9)
        s = csg.TypedSlots(7, selector="sine_sigmoid_multi")
        r = ref.PrimitivePool(ref.PoolCfg(n_leaves=7, mode="sine_sigmoid_multi"))
        r.load_state_dict(s.state_dict())
        for beta in (1.0, 20.0):
            self.assertTrue(torch.equal(s(self.x, beta, "soft"), r.get_occs(self.x, beta, hard_leaf=False)))
            self.assertTrue(torch.equal(csg.multihot_mask(s.type_logits, beta, "sine_sigmoid_multi", 0.5),
                                        sweep.type_multihot_mask(r, beta, 0.5)))


class TestAssemblers(unittest.TestCase):
    def setUp(self):
        self.q = torch.sigmoid(rand(12, 40, seed=10) * 3)

    def test_dnf(self):
        m = ref.NaryDNFModel(12, 5, "sine_softmax", "product", True, 10.0)
        with torch.no_grad():
            m.m_pos.copy_(rand(12, 5, seed=11) * 2); m.m_neg.copy_(rand(12, 5, seed=12) * 2); m.act.copy_(rand(5, seed=13))
        for hard in (False, True):
            self.assertTrue(torch.equal(csg.nary_dnf(self.q, m.m_pos, m.m_neg, m.act, hard),
                                        ref.dnf_forward(self.q, m.m_pos, m.m_neg, m.act, hard_bool=hard)))

    def test_stump(self):
        m = ref.ProperCSGStumpModel(12, 5, "sine_softmax", "product", True, 10.0)
        with torch.no_grad():
            m.comp.copy_(rand(12, seed=14)); m.inter.copy_(rand(12, 5, seed=15) * 2); m.union.copy_(rand(5, seed=16))
        for hard in (False, True):
            self.assertTrue(torch.equal(csg.csg_stump(self.q, m.comp, m.inter, m.union, hard),
                                        m._stump_forward(self.q, hard_bool=hard)))

    def test_stump_minmax(self):
        m = ref.PaperStumpModel(12, 5, "sine_softmax", "product", True, 10.0, 20.0)
        with torch.no_grad():
            m.wc.copy_(rand(12, seed=17)); m.wi.copy_(rand(12, 5, seed=18) * 2); m.wu.copy_(rand(5, seed=19))
        self.assertTrue(torch.equal(csg.csg_stump_minmax(self.q, m.wc, m.wi, m.wu, False, 20.0), m._stump(self.q, hard_bool=False)))
        self.assertTrue(torch.equal(csg.csg_stump_minmax(self.q, m.wc, m.wi, m.wu, True), sweep.paper_stump_exact_hard(m, self.q, 0.5)))

    def test_tree(self):
        for ops in ("csg3", "csg5"):
            m = ref.BinaryTreeModel(12, "sine_softmax", "product", True, 10.0, ops)
            with torch.no_grad():
                m.gates.copy_(rand(12, seed=20)); m.op_logits.copy_(rand(*m.op_logits.shape, seed=21))
            for hard in (False, True):
                self.assertTrue(torch.equal(csg.binary_tree(self.q, m.gates, m.op_logits, hard),
                                            m._tree_reduce(self.q, hard_bool=hard)), (ops, hard))


if __name__ == "__main__":
    unittest.main()
