# PySIGN: Science Informed Graph Networks

### Key Features
- **Complete Resources**: We incorporate diverse science-informed data resources ranging from physics to biochemistry;
- **Widely-covered Models**: We involve state-of-the-art science-informed graph neural networks across a wide domain;
- **Uniform Pipelines**: We formulate a uniform and extensible pipeline for training and evaluating science-informed graph networks;
- **Elaborated Toolkits**: We provide useful toolkits for directly analyzing the predictions generated by the models.

![Design](assets/pysign.png "Design")

### Requirements
```
torch==1.7.1
torch_scatter==2.0.7
torch_sparse==0.6.9
torch_cluster==1.5.8
torch-geometric==2.0.4
tqdm
matplotlib
sympy
pyyaml
lie-learn
atom3d
```
You can also use the `Dockerfile` in `docker` folder to build the environment.

### Quick Start
TBD

### Models
#### Encoder

We implement several following equivariant geometric networks as encoders.

##### Irreducible Representation

[TFN](https://arxiv.org/pdf/1802.08219), [SE(3)-Transformer](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)

##### Scalarization

[SchNet](https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf), [DimeNet](https://arxiv.org/pdf/2003.03123), [Radial Field (RF)](https://arxiv.org/pdf/1910.00753), [EGNN](http://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf), [GMN](https://openreview.net/forum?id=SHbhHHfePhP), [PaiNN](http://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf), [Equivariant Transformer (ET)](https://openreview.net/pdf?id=zNHzqZ9wrRB)

All supported encoders are registered in the class ``EncoderRegistry``. For example, one can easily instantiate an EGNN as follows:

```python
from pysign.nn.model import EncoderRegistry

encoder = EncoderRegistry.get_encoder('EGNN')
model = encoder(in_node_nf=10, hidden_nf=128, out_node_nf=128, in_edge_nf=0, n_layers=2)
```

#### Decoder

Decoders are applied to transform the encoded scalar (and/or) vector representations into target outputs. One can construct different types of decoders by specifying the parameters of ``GeneralPurposeDecoder``. For example, the following codes will produce a decoder which generates a vector output for each node (``target='vector'``) by firstly predicting a global scaler via an MLP (``decoding='MLP'``) and then calculating the gradients of the scalar w.r.t. the node positions, in order to acquire the vector outputs (``vector_method='gradient'``). The parameter ``dynamics`` can be optionally set to True if the task features dynamics prediction.

```python
from pysign.nn.model import GeneralPurposeDecoder

decoder = GeneralPurposeDecoder(hidden_dim=128, output_dim=1, decoding='MLP', target='vector',
                                vector_method='gradient', dynamics=True)
```

We support different modes for ``decoding`` and ``vector_method`` to choose the required decoders. The created encoder and decoder should satisfy the following table.

#### Matched Encoder-Decoder for Vector-valued Predictions

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
	              decoding='MLP', vector_method=None, scalar_pooling='sum', target='scalar', 
	              return_outputs=False)
```

Meanwhile, one can also conduct a dynamics prediction task by switching the parameters, which returns vector features for each node.

```python
task = Prediction(rep=model, output_dim=1, rep_dim=128, task_type='Regression', loss='MAE',
                  decoding='MLP', vector_method='gradient', target='vector', dynamics=True, 
                  return_outputs=True)
```

The contrastive task predicts the difference for multiple 3D graphs.

```python
from pysign.task import Contrastive

task = Contrastive(rep=model, output_dim=1, rep_dim=128, task_type='BinaryClassification', 
	               loss='BCE', return_outputs=True, dynamics=False)
```

### Datasets & Benchmarks

We currently support 4 benchmarks with various tasks.

[QM9](https://www.nature.com/articles/sdata201422) is a small molecule datasets containing 134k 3D molecules and 12 tasks to predict the geometric, energetic, electronic, and thermodynamic properties for the molecules.

[MD17](https://www.science.org/doi/full/10.1126/sciadv.1603015) calculates the molecular dynamics trajectories for 8 small molecules. Based on previous works, we construct 2 benchmarks on MD17. The energy & force prediction task predicts the energy for the whole molecule and the force for each atom, and the dynamics prediction task is required to generate the MD trajectory given the initial state.

[Atom3D](https://www.atom3d.ai/) designs 8 tasks for 3D biomolecules, like small molecules, proteins, and nucleic acids. We currently focus on 2 of them, named LBA and LEP.  LBA is a prediction task which predicts the binding affinity of a protein pocket and a ligand, and LEP is a contrastive task to predict whether the small molecule will activate the protein’s function or not.

[N-body](https://arxiv.org/abs/1802.04687) is a simulation dataset depicting the dynamics trajectories for several charged particals in a physical system. The task is to predict the trajectory given an initial state.

The summary of currently available datasets and corresponding benchmarks are provided below:

| Datasets         | Benchmarks                                         |
| ---------------- | -------------------------------------------------- |
| ``QM9``          | ``benchmark_qm9``                                  |
| ``MD17``         | ``benchmark_md17``                                 |
| ``MD17Dynamics`` | ``benchmark_md17_dynamics``                        |
| ``Atom3D``       | ``benchmark_atom3d_lba``, ``benchmark_atom3d_lep`` |
| ``NBody``        | ``benchmark_nbody_dynamics``                       |

All supported benchmarks are registered in ``BenchmarkRegistry``.  For example, one can launch a QM9 benchmark as follows.

```shell
python examples/run_benchmark.py -b benchmark_qm9
```

### Visualization

We provide visualization guidelines in ``visualization`` module.

#### Trajectories on MD17

![MD17](assets/md17_dynamics.gif "MD17")

#### Trajectories on N-body

![NBody](assets/nbody.gif "NBody")


### Reference

### Core Developers

Jiaqi Han: hanjq21@mails.tsinghua.edu.cn

Rui Jiao: jiaor21@mails.tsinghua.edu.cn

The codebase is currently under active development!

