# IRENE
## Dependency
Python == 3.8.20

GCC 9.4.0

Rustc 1.81.0

Driver Version: 570.133.20     

CUDA Version: 12.8

Linux

conda 23.1.0

Run the following command in the root directory of this repo:
```
conda env create -f environment.yml
```
## Inference
### Datesets
#### Demonstration corpus

You can construct the demonstration corpus from your own training dataset, following the format below (Reference Only).
```
{
    "idx": 0,
    "c_code": "#include...",
    "rust_code": "use std::io::...",
    "compile_result": "Compile Success"
}
```
After that, you can use the following command to run the demonstration corpus that has been categorized by the Static Rule Analyzer.
```
python static_analyzer.py
```
#### Evaluation dataset

`./dataset/groundtruth`， We extracted different C and Rust solutions to the same problem from the evaluation set of [ntunlp/xCodeEval](https://github.com/ntunlp/xCodeEval.git), and further filtered them to ensure each pair passes all test cases.

You may also use your own data. The recommended format is as follows (Reference Only):
```
{
    "source_code": "#include..."
}
```

We recommend using the `jsonline` format. However, if your data consists of many separate `json` files like those in `./dataset/groundtruth`, we also provide different `load_data()` functions to support that.
### Usage
#### Generate code summary
1. Modify`./infer.sh`: excute this file: `./gen_summary.py`，and the model used `--model_name`
2. You can use the following command
```
# if use local models:
./infer.sh local
# if use api
./infer.sh api
```
#### Generate translation
1. Modify`./infer.sh`: excute this file: `./translation.py`，and the model and method used `--model_name ${model} --method`, method must be one of: IRENE, Instruction, ICL, RAG, COT, Vert.
2. You can use the following command
```
# if use local models:
./infer.sh local
# if use api
./infer.sh api
```
#### Feedback generation
1. Modify`./infer.sh`: excute this file: `./Feedback.py`，and the model and method used `--model_name ${model} --method`, method must be one of: IRENE, Instruction, ICL, RAG, COT, Vert.
2. You can use the following command
```
# if use local models:
./infer.sh local
# if use api
./infer.sh api
```
## Evaluation
Dependency and other information please refer to [ntunlp/xCodeEval](https://github.com/ntunlp/xCodeEval.git)
1. Run the following command to use the docker
```
./manage_docker.sh start
```
2. Modify the path in `./my_eval` and run:
```
python my_eval.py
```

## Citation

Part of the data and `my_eval.py` is adapted from [ntunlp/xCodeEval](https://github.com/ntunlp/xCodeEval.git). 
Please check their repository for more details and original license. Thanks to the authors for their excellent work.