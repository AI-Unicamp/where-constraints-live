from __future__ import annotations

from dataclasses import dataclass

from .shapes import TargetShape, l_shape, normalize_targets, polyomino_from_ascii, rectangle
from .transforms import all_transforms, transform_then_translate


@dataclass(frozen=True)
class Benchmark:
    name: str
    targets: list[TargetShape]
    ks: list[int]
    expected: str


def identical_translated() -> Benchmark:
    base = l_shape()
    targets = normalize_targets([base, frozenset((x + 10, y + 4) for x, y in base)], ["A", "B"])
    return Benchmark("identical_translated", targets, [1, 2, 6], "rigid reuse should recover stock=area")


def identical_rotated() -> Benchmark:
    base = l_shape()
    rot = all_transforms(2, True)[5]
    targets = normalize_targets([base, transform_then_translate(base, rot, (8, 1))], ["A", "B"])
    return Benchmark("identical_rotated", targets, [1, 2, 6], "rigid reuse should recover rotated stock=area")


def overlay_wins() -> Benchmark:
    a = rectangle(0, 0, 4, 3)
    b = frozenset(set(rectangle(0, 0, 4, 3)) - {(3, 2)} | {(4, 2)})
    targets = normalize_targets([a, b], ["A", "B"])
    return Benchmark("overlay_wins", targets, [2, 3, 5], "positional overlap is intentionally strong")


def positive_trash_budget() -> Benchmark:
    line = frozenset((0, y) for y in range(7))
    hook = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 3)})
    targets = normalize_targets([line, hook], ["line", "hook"])
    return Benchmark("positive_trash_budget", targets, [3, 4, 7], "K=3 needs positive trash; K=4 reaches zero trash")


def local_motif_permutation() -> Benchmark:
    motif_l = frozenset({(0, 0), (1, 0), (0, 1)})
    motif_i = frozenset({(0, 0), (1, 0), (2, 0)})
    a = frozenset(set(transform_then_translate(motif_l, all_transforms(2)[0], (0, 0))) | set(transform_then_translate(motif_i, all_transforms(2)[0], (5, 0))))
    b = frozenset(set(transform_then_translate(motif_i, all_transforms(2)[1], (0, 5))) | set(transform_then_translate(motif_l, all_transforms(2)[3], (5, 5))))
    targets = normalize_targets([a, b], ["A", "B"])
    return Benchmark("local_motif_permutation", targets, [2, 3, 6], "motifs are shared but not positionally aligned")


def three_shape_signature() -> Benchmark:
    square = frozenset({(0, 0), (1, 0), (0, 1), (1, 1)})
    i_tet = frozenset({(0, 0), (1, 0), (2, 0), (3, 0)})
    t_tet = frozenset({(0, 0), (1, 0), (2, 0), (1, 1)})
    l_tet = frozenset({(0, 0), (0, 1), (0, 2), (1, 0)})
    tf = all_transforms(2, True)
    a = frozenset(
        set(transform_then_translate(square, tf[0], (0, 0)))
        | set(transform_then_translate(i_tet, tf[0], (10, 0)))
        | set(transform_then_translate(t_tet, tf[0], (20, 0)))
    )
    b = frozenset(
        set(transform_then_translate(square, tf[5], (2, 8)))
        | set(transform_then_translate(i_tet, tf[4], (12, 8)))
        | set(transform_then_translate(l_tet, tf[1], (22, 8)))
    )
    c = frozenset(
        set(transform_then_translate(square, tf[3], (4, 16)))
        | set(transform_then_translate(t_tet, tf[2], (14, 16)))
        | set(transform_then_translate(l_tet, tf[6], (24, 16)))
    )
    targets = normalize_targets([a, b, c], ["A", "B", "C"])
    return Benchmark("three_shape_signature", targets, [4], "known all-shared plus pair-shared signature structure")


def no_reuse() -> Benchmark:
    a = polyomino_from_ascii(
        """
        ####
        #...
        ###.
        """
    )
    b = polyomino_from_ascii(
        """
        #..
        ###
        ..#
        ..#
        """
    )
    # both have 8 cells after adding one far cell to b
    b = frozenset(set(b) | {(6, 0)})
    targets = normalize_targets([a, b], ["A", "B"])
    return Benchmark("no_reuse", targets, [2, 4, 8], "solver should not invent unsupported sharing")


def fast_benchmarks() -> list[Benchmark]:
    return [
        identical_translated(),
        identical_rotated(),
        overlay_wins(),
        positive_trash_budget(),
        local_motif_permutation(),
        three_shape_signature(),
    ]


def full_benchmarks() -> list[Benchmark]:
    return fast_benchmarks() + [no_reuse()]
