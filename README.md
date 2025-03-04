# LLMOpt


This the code for LLMOpt: Query Optimization utilizing Large Language Models.

# Requirements

1. Prepare PostgreSQL
You can find corresponding version of PostgreSQL in (https://www.postgresql.org/ftp/source/v16.1/)

```sh
cd postgresql-16.1/
./configure --prefix=/usr/local/pgsql --without-icu
make
make install
```

2. Install *pg_hint_plan*. 
We modifie it with [pg_hint_plan_lucifer] (https://github.com/lucifer12346/pg_hint_plan_lucifer)

```sh
cd ./pg_hint_plan_lucifer/
make
make install
```

3. Python Requirements
For LLM Training, we use:

```
pytorch  2.4.1+cu118 
torchaudio    2.4.1+cu118 
torchvision    0.19.1+cu118 
transformers    4.44.2
accelerate     0.30.1
deepspeed    0.14.4
datasets     2.21.0 
tensorboard 2.14.0
math
tpdm

```
in python 3.8.5

For LLM inference, we use:
```
vllm==0.6.3.post1
```
in python 3.9.0


The environment varies in different hardwares.



# Utilizing An LLM As Generator

1. Training
You have to modify *YOUR_ACCELERATE_CONFIG* and *PRETRAINED_MODEL* in the script to your environment config and model path.
```
cd scripts
bash train_sel.bash
```


2. Inference
You have to modify *ckpt* in the script.

```
bash infer_sel.bash
```

3. We use Tree-CNN in [BAO](https://github.com/learnedsystems/baoforpostgresql) to select the optimal hint from "pred_hints".




# Utilizing An LLM As Selector

1. Following [BAO](https://rmarcus.info/bao_docs/introduction.html), we use the most common hints to prepare data.

2. Training
You have to modify *YOUR_ACCELERATE_CONFIG* and *PRETRAINED_MODEL* in the script to your environment config and model path.
```
cd scripts
bash train_gen.bash
```


3. Inference
You have to modify *ckpt* in the script.

```
bash infer_gen.bash
```

The hint in "pred_hints" is the chosen one
