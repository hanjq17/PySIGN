# PySIGN: Science Informed Graph Networks

PySIGN Team,  Tsinghua University

1. Complete resources.
2. Widely-covered models.
3. Uniformed pipelines.
4. Elaborated toolkits.

![Design](assets/pysign.png "Design")

### Encoder

We implement several following equivariant geometric networks as encoders.

#### Irreducible Representation

[TFN](https://arxiv.org/pdf/1802.08219), [SE(3)-Transformer](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)

#### Scalarization

[SchNet](https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf), [DimeNet](https://arxiv.org/pdf/2003.03123), [Radius Field (RF)](https://arxiv.org/pdf/1910.00753), [EGNN](http://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf), [PaiNN](http://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf), [Equivariant Transformer (ET)](https://openreview.net/pdf?id=zNHzqZ9wrRB)

All supported encoders are registered in ``EncoderRegistry``. For example, one can create an EGNN model as follows.

```python
from pysign.nn.model import EncoderRegistry

encoder = EncoderRegistry.get_encoder('EGNN')
model = encoder(in_node_nf=10, hidden_nf=128, out_node_nf=128, in_edge_nf=0, n_layers=2)
```

### Decoder

Decoders are applied to transform the encoded scalar & vector representations into target outputs. One can construct diffenent types of decoders by switching the mode parameters of ``GeneralPurposeDecoder``. For example, the following codes generate a decoder which generates a vector output for each node (``dynamics=True``) by firstly predicting a global scaler via an MLP (``decoding='MLP'``) and secondly calculating the gradients of the node positions w.r.t. the global scaler to acquire the vector outputs (``vector_method='gradient'``).

```python
from pysign.nn.model import GeneralPurposeDecoder

decoder = GeneralPurposeDecoder(hidden_dim=128, output_dim=1, decoding='MLP',
                                vector_method='gradient', dynamics=True)
```

We support different modes for ``decoding`` and ``vector_method`` to choose the required decoders. The created encoder and decoder should satisfy the following table.

### Matched Encoder-Decoder for Dynamics Prediction

|                   | MLP+diff | MLP+gradient | GatedBlock |
| ----------------- | -------- | ------------ | ---------- |
| TFN               | &#10004; |              |            |
| SE(3)-Transformer | &#10004; |              |            |
| RF                | &#10004; |              |            |
| EGNN              | &#10004; | &#10004;     |            |
| SchNet            |          | &#10004;     |            |
| DimeNet           |          | &#10004;     |            |
| PaiNN             |          | &#10004;     | &#10004;   |
| ET                |          | &#10004;     | &#10004;   |

### Task

We support 2 types of tasks in general, named ``Prediction`` and ``Contrastive``.

The prediction task predicts the scaler or vector features given a single 3D graph. Take molecular property prediction as an example, it is a regression problem requiring a real number for the total graph.

```python
from pysign.task import Prediction

task = Prediction(rep=model, output_dim=1, rep_dim=128, task_type='Regression', loss='MAE',
                    decoding='MLP', vector_method=None, scalar_pooling='sum', target='scalar', return_outputs=False)
```

Meanwhile, one can also conduct a dynamics prediction task by switching the parameters, which returns vector features for each node.

```python
task = Prediction(rep=model, output_dim=1, rep_dim=128, task_type='Regression', loss='MAE',
                    decoding='MLP', vector_method='gradient', target='vector', dynamics=True, return_outputs=True)
```

The contrastive task predicts the difference for multiple 3D graphs.

```python
from pysign.task import Contrastive

task = Contrastive(rep=model, output_dim=1, rep_dim=128, task_type='BinaryClassification', loss='BCE',
                    return_outputs=True, dynamics=False)
```

### Dataset & Benchmark

We currently support 4 benchmarks with various tasks.

[QM9](https://www.nature.com/articles/sdata201422) is a small molecule datasets containing 134k 3D molecules and 12 tasks to predict the geometric, energetic, electronic, and thermodynamic properties for the molecules.

[MD17](https://www.science.org/doi/full/10.1126/sciadv.1603015) calculates the molecular dynamics trajectories for 8 small molecules. Based on previous works, we construct 2 benchmarks on MD17. The energy & force prediction task predicts the energy for the whole molecule and the force for each atom, and the dynamics prediction task is required to generate the MD trajectory given the initial state.

[Atom3D](https://www.atom3d.ai/) designs 8 tasks for 3D biomolecules, like small molecules, proteins, and nucleic acids. We currently focus on 2 of them, named LBA and LEP.  LBA is a prediction task which predicts the binding affinity of a protein pocket and a ligand, and LEP is a contrastive task to predict whether the small molecule will activate the protein’s function or not.

N-body is a simple dataset simulating the dynamics trajectories for several charged particals in a physical system. The task is to predict the trajectory given an initial state.

All supported benchmarks are registered in ``BenchmarkRegistry``.  For example, one can launch a QM9 benchmark as follows.

```shell
python examples/run_benchmark.py -b benchmark_qm9
```

### Visualization

#### Trajectories on MD17

![MD17](assets/md17_dynamics.gif "MD17")

#### Trajectories on N-body

![NBody](assets/nbody.gif "NBody")

### Core Developers

Jiaqi Han: hanjq21@mails.tsinghua.edu.cn

Rui Jiao: jiaor21@mails.tsinghua.edu.cn
