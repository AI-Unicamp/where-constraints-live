# Where Constraints Live

Code for the MSc dissertation

> Vinicius Hirono Gonçalves. *Where Constraints Live: Representation in Inverse CSG and Geometric Dissection*.
> MSc dissertation, FEEC, Universidade Estadual de Campinas (UNICAMP), 2026. Advisor: Paula Dornhofer Paro Costa.

The repository has two parts:

* **`wcl/`** — a small library with the methods of the dissertation, written to be read and reused;
* **`reproduce/`** — the scripts that produced every table and figure, with the exact commands.

Every function in `wcl/` is tested against the code that produced the dissertation's numbers
(`tests/`); the tests check exact equality, not approximate agreement.

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, torch, shapely, scikit-image, libigl, trimesh, ...
python -m unittest discover tests      # about one minute
```

Python 3.10 or newer. The C++ lattice searches (`reproduce/dissection/experiments/zhou2d.cpp`, `zhou_multi.cpp`,
`reproduce/dissection/zhou3d/zhou3d.cpp`) are single files with no dependency besides the public-domain
`stb_image` headers in `reproduce/dissection/zhou3d/`; build them with any C++17 compiler, e.g.
`g++ -std=c++17 -O3 -pthread zhou3d.cpp -o zhou3d`.

## Library

### `wcl.csg` — inverse CSG with typed slots (Chapter 3)

```python
import torch, wcl.csg as csg

slots = csg.TypedSlots(50, selector="sine_softmax")   # 50 slots, each holding all 5 primitive types
x = torch.rand(4096, 3) * 2 - 1                        # query points in [-1, 1]^3
q = slots(x, beta=20.0, extraction="multihot")         # (50, 4096) slot occupancies
pos, neg, act = torch.randn(50, 16), torch.randn(50, 16), torch.randn(16)
o = csg.nary_dnf(q, pos, neg, act, hard=True)          # occupancy of the extracted CSG program
```

| function | what it does |
|---|---|
| `primitive_sdf`, `sphere_sdf`, `box_sdf`, `cylinder_sdf`, `wedge_sdf`, `corner_wedge_sdf` | signed distances of the five primitive types |
| `volume_radius`, `primitive_volume` | volume-equivalent radius used to calibrate the fields (see note below) |
| `to_local`, `euler_to_rotation`, `sdf_to_occupancy` | primitive frames and calibrated soft occupancy |
| `type_weights`, `combine_types`, `onehot_occupancy`, `multihot_mask` | typed-slot competition and one-hot / multi-hot extraction |
| `nary_dnf`, `csg_stump`, `csg_stump_minmax`, `binary_tree` | the four Boolean assemblers compared in the dissertation |
| `TypedSlots` | learnable slots (torch module) combining the above |

*Note on wedge volumes.* The dissertation's runs normalise wedges by the volume proxies `4abc` (wedge) and `2abc`
(corner wedge); the exact volumes of these solids are `7abc` and `19/3 abc`. `volume_radius` reproduces the
runs by default and gives the exact volumes with `exact=True`.

### `wcl.dissection` — geometric dissection (Chapter 5)

```python
from wcl.dissection import lattice, baco

lb, isos = lattice.match_labels(labels_a, labels_b)   # pair lattice pieces, one isometry each
prob = baco.problem(labels_a, lb, isos, sorted(isos))
loops = prob.unpack(baco.solve(prob))                 # one shared boundary loop per piece (2D)
```

| module | what it does |
|---|---|
| `lattice` | the 8 lattice isometries, best isometry per piece pair, Hungarian matching of pieces |
| `baco` | BACO in 2D: shared loops, contact / exterior / fairness residuals, robust least squares |
| `patch3d` | the 3D shared patch complex: construction (`complex`), contact closure as shared variables (`closure`), least-squares system (`system`), untangling (`intersections`), winding-number certificate (`certificate`); `python -m wcl.dissection.patch3d.pipeline --help` runs the whole method |

## Reproducing the dissertation

See [`reproduce/README.md`](reproduce/README.md): one command per table, the settings of every run, and which
inputs come from the data deposit.

## Data

Result data and the inputs needed to rerun the experiments are deposited separately (Zenodo / REDU; DOIs in the
dissertation and in `CITATION.cff`).

## Third-party inputs

The 3D meshes of Chapter 3 (Stanford Bunny, Fandisk, Rocker-arm) are downloaded by the scripts from
[`alecjacobson/common-3d-test-models`](https://github.com/alecjacobson/common-3d-test-models). The 2D silhouettes
of the comparison with Duncan et al. (2017) are extracted from their paper, which is not redistributed. The
winding numbers and signed distances use [libigl](https://libigl.github.io).

## License

MIT (see `LICENSE`). If you use this code, please cite the dissertation (`CITATION.cff`).
