# Category-Held-Out Surface Primitive and Boundary Parsing

| | |
| --- | --- |
| Final rank | #31 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-09-11 |

## Problem statement

### Overview

Label an unorganized 3D point cloud with both its surface primitive and its proximity to a geometric boundary. Generalize to CAD categories absent from training, using coordinates rather than supplied normals, CAD topology or axis labels.

### Dataset

- `train.csv` / `test.csv`: `task_id`; 700 training objects and 200 test objects.
- `train_points.npz` / `test_points.npz`: arrays keyed by `task_id`, with `N × 3` coordinates and at most 1,024 points per object.
- `train_labels.csv` / `sample_submission.csv`: `task_id,target_json`.

All objects in a source category remain in one split, including template-generated variants. Exact normalized-coordinate duplicates are removed. Each cloud is centered, scaled to unit maximum radius, independently rotated and point-shuffled. Subsampling selects native points without interpolation. Only source coordinates are inputs; primitive and boundary annotations remain targets. Similar local geometry can recur across categories, and sparse sampling makes some boundaries ambiguous.

### Task

Return one integer per point, in the supplied array order: `label = 2 × primitive + near_edge`. Primitive codes are 0 other, 1 cone, 2 cylinder and 3 plane. `near_edge` is 0 or 1, giving eight joint classes.

Objects are selected by fixed hash order within the existing category-held-out partitions. Each selected cloud retains a deterministic 1,024-point subset (or all points if fewer), with the matching primitive and boundary labels at exactly those indices. No labels are interpolated.

### Submission

```
task_id,target_json
example_1,"[6,7,4,4]"
example_2,"[2,3,0]"
```

These abbreviated examples illustrate variable point counts. Actual list length must equal the object's point count. Columns must be exactly `task_id,target_json`, in that order, with each test ID once. Wrong-length lists, noninteger/out-of-range labels or malformed JSON give that object zero.

### Evaluation

Mean per-object intersection-over-union. Within each object, compute `TP / (TP + FP + FN)` for every joint class present in either prediction or reference, then average those class IoUs. Average the resulting scores equally across objects. Score 1 is perfect. This balances joint geometric classes within each object without letting larger clouds dominate. Malformed rows receive zero, not an abstention that can improve another object's score. File-level schema and ID errors are rejected.

### Expected Methods

Point-cloud networks and learned local geometric descriptors. Generic public pretrained models are allowed.

### What Not To Use

No original CAD/mesh lookup, source point labels, source-specific checkpoints, external labeled CAD data, manual test annotation or hosted APIs.

**Compute Environment**
 The competition runtime provides one NVIDIA A10G GPU.
 The maximum end-to-end runtime is 1.5 hours, including data loading, training or adaptation, inference, structured decoding, validation, and submission writing.
