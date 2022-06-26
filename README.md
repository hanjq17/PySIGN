# AirGeom: A Universal Toolkit for Geometric Graphs

AirGeom Team, Institute for AI Industry Research (AIR), Tsinghua University

1. Complete resources.
2. Widely-covered models.
3. Uniformed pipelines.
4. Elaborated toolkits.

### Matched Enc-Dec for MD

|                | Scalar   | EquivariantVector | DifferentialVector |
| -------------- | -------- | ----------------- | ------------------ |
| TFN            |          |                   | &#10004;           |
| SE3Transformer |          |                   | &#10004;           |
| RF             |          |                   | &#10004;           |
| EGNN           | &#10004; |                   | &#10004;           |
| SchNet         | &#10004; |                   |                    |
| PaiNN          | &#10004; | &#10004;          |                    |
| ET             | &#10004; | &#10004;          |                    |

### Trajectories on MD17

![GT](visualization/vis/Aspirin_GT_1000.gif "GT")
![EGNN](visualization/vis/Aspirin_EGNN_DifferentialVector_1000.gif "EGNN")


### Trajectories on NBody

![EGNN](visualization/vis/animation_10.gif "EGNN")
![EGNN](visualization/vis/animation_15.gif "EGNN")
![EGNN](visualization/vis/animation_20.gif "EGNN")