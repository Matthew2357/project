# Investigating LoRA Aggregation in Federated Learning

This is the code base for my semester research project. The paper can be found in ```report```. This codebase is built on top of https://github.com/epfml/personalized-collaborative-llms.git by Nicolas Wagner.
## Quickstart

Install conda environment:

```
conda env create -f env.yml && conda activate llm-lora
```

### Reproducibility

To reproduce the experiments from the paper in the homogeneous-ranks scenario for the datasets Slim Pajama and Wikipedia Multilingual, please run the command

```./scripts/reproduceScript.sh```

This will take a while to run, as there are 3 seeds, 6 methods, 6 values of Dirichlet-alpha and 2 datasets.


