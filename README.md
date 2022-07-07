# PySIGN: Science Informed Graph Networks

PySIGN Team,  Tsinghua University

1. Complete resources.
2. Widely-covered models.
3. Uniformed pipelines.
4. Elaborated toolkits.

![Design](assets/pysign.png "Design")





### Encoder

xxx scalar/ irreps



### Decoder

decoder_module and operation (diff / gradient)



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



### Task

Prediction / contrastive

Prediction: multitask, multiple evaluation metrics



### Dataset & Benchmark

QM9: xxx,  -> prediction task









### Trajectories on MD17

![MD17](assets/md17_dynamics.gif "MD17")


### Trajectories on NBody

![NBody](assets/nbody.gif "NBody")


### Core Developers

Jiaqi Han: hanjq21@mails.tsinghua.edu.cn

Rui Jiao: jiaor21@mails.tsinghua.edu.cn