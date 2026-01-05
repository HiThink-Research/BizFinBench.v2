<p align="center">
  <h1 align="center">
    <img src="static/logo.png" alt="BizFinBench logo" height="40" style="position:relative; top:6px;">
  BizFinBench.v2: A Unified Offline–Online Bilingual Benchmark for Expert-Level Financial Capability Evaluation of LLMs</h1>
    <p align="center">
    <span class="author-block">
      Xin Guo<sup>1,*</sup> </a>,</span>
                <span class="author-block">
      Rongjunchen Zhang<sup>1,*,♠</sup></a>, Guilong Lu<sup>1</sup>, Xuntao Guo<sup>1</sup>, Jia Shuai<sup>1</sup>, Zhi Yang<sup>1</sup>, Liwen Zhang<sup>1,♠</sup>
    </span>
    </div>
    <div class="is-size-5 publication-authors" style="margin-top: 10px;">
        <span class="author-block">
            <sup>1</sup>Hithink Research, <sup>2</sup>Harbin Institute of Technology
        </span>
        <br>
        <span class="author-block">
            <sup>*</sup>Co-first authors, <sup>♠</sup>Corresponding author, zhangrongjunchen@myhexin.com
        </span>
    </div>
  </p>
  <p>
  📖<a href="https://arxiv.org/abs/2505.19457">Paper</a> |🏠<a href="https://hithink-research.github.io/BizFinBench/">Homepage</a></h3>|🤗<a href="https://huggingface.co/datasets/HiThink-Research/BizFinBench">Huggingface</a></h3>
  </p>
<div align="center"></div>
<p align="center">

Large language models excel across general tasks, yet judging their reliability in logic‑heavy, precision‑critical domains such as finance, law and healthcare is still difficult. To address this challenge, we propose **BizFinBench**, the first benchmark grounded in real-world financial applications. **BizFinBench** comprises over 100,000+ bilingual (English & Chinese) financial questions, each rooted in real-world business scenarios. The first public release, **BizFinBench.v1**, delivers 6,781 well annotated Chinese queries, covering five dimensions: numerical calculation, reasoning, information extraction, prediction recognition and knowledge‐based question answering, which are mapped to nine fine-grained categories.

## 📢 News 
- 🚀 [06/01/2026] TBD


## 💡 Highlights
- 🔥  **Benchmark:** We propose **BizFinBench**, the first evaluation benchmark in the financial domain that integrates business-oriented tasks, covering 5 dimensions and 9 categories. It is designed to assess the capacity of LLMs in real-world financial scenarios.

## 📕 Data Distrubution
This dataset contains multiple subtasks, each focusing on a different financial understanding and reasoning ability, as follows:

<img src="static/distribution.png" alt="Data Distribution">

## 📚 Example
<img src="static/Anomalous Event Attribution.drawio.png" alt="Data Distribution">


## 🛠️ Usage

### Contents

```
llm-eval
├── README.md
├── benchmark_code
├── config # All custom sample configs can be found in this folder
├── envs  #env settings
├── inference # All inference-engine-related code is in this folder
├── post_eval.py # Evaluation launcher after inference is finished
├── reqirements.txt
├── run.py # Entry point for the entire evaluation workflow
├── run.sh # Sample execution script for launching an evaluation; maintain your own run.sh as needed
├── scripts # Reference run.sh scripts
├── tools # tools
├── statistic.py # Aggregates final evaluation statistics
└── utils
```

### Install requirements
```sh
pip install -r requirements.txt
```
### Quick Start – Evaluate a Local Model

```sh
export MODEL_PATH=model/Qwen2.5-0.5B   # Path to the model to be evaluated
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template   # Hithink llama3 llama2 none qwen chat_template; chat_template is recommended

# First start the model as a service
python inference/predict_multi_gpu.py \
    --model ${MODEL_PATH} \
    --server_port ${REMOTE_MODEL_PORT} \
    --prompt ${PROMPT_TYPE} \
    --preprocess preprocess \
    --run_forever \
    --max_new_tokens 4096 \
    --tensor_parallel ${TENSOR_PARALLEL} & 

# Pass in the config file path to start evaluation
python run.py --config config/offical/eval_fin_eval_diamond.yaml --model_name ${MODEL_NAME}
```

### Quick Start – Evaluate a Local Model and Score with a Judge Model

```sh
export MODEL_PATH=model/Qwen2.5-0.5B   # Path to the model to be evaluated
export REMOTE_MODEL_PORT=16668
export REMOTE_MODEL_URL=http://127.0.0.1:${REMOTE_MODEL_PORT}/model
export MODEL_NAME=Qwen2.5-0.5B
export PROMPT_TYPE=chat_template   # llama3 llama2 none qwen chat_template; chat_template is recommended

# First start the model as a service
python inference/predict_multi_gpu.py \
    --model ${MODEL_PATH} \
    --server_port ${REMOTE_MODEL_PORT} \
    --prompt ${PROMPT_TYPE} \
    --preprocess preprocess \
    --run_forever \
    --max_new_tokens 4096 \
    --tensor_parallel ${TENSOR_PARALLEL} \
    --low_vram & 

# Start the judge model
export JUDGE_MODEL_PATH=/mnt/data/llm/models/base/Qwen2.5-7B
export JUDGE_TENSOR_PARALLEL=1
export JUDGE_MODEL_PORT=16667
python inference/predict_multi_gpu.py \
    --model ${JUDGE_MODEL_PATH} \
    --server_port ${JUDGE_MODEL_PORT} \
    --prompt chat_template \
    --preprocess preprocess \
    --run_forever \
    --manual_start \
    --max_new_tokens 4096 \
    --tensor_parallel ${JUDGE_TENSOR_PARALLEL} \
    --low_vram &

# Pass in the config file path to start evaluation
python run.py --config "config/offical/eval_fin_eval.yaml" --model_name ${MODEL_NAME}
```

> **Note**: Add the `--manual_start` argument when launching the judge model, because the judge must wait until the main model finishes inference before starting (this is handled automatically by the `maybe_start_judge_model` function in `run.py`).

### Quick Start – Evaluate external apis (e.g., chatgpt)

```sh
export API_NAME=chatgpt # The api name, currently support chatgpt
export API_KEY=xxx # Your api key
export MODEL_NAME=gpt-4.1

# Pass in the config file path to start evaluation
python run.py --config config/offical/eval_fin_eval_diamond.yaml --model_name ${MODEL_NAME}
```
> **Note**: You can adjust the API’s queries-per-second limit by modifying the semaphore_limit setting in envs/constants.py. e.g., GPTClient(api_name=api_name,api_key=api_key,model_name=model_name,base_url='https://api.openai.com/v1/chat/completions', timeout=600, semaphore_limit=5)

## ✒️Citation

```
@article{lu2025bizfinbench,
  title={BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs},
  author={Lu, Guilong and Guo, Xuntao and Zhang, Rongjunchen and Zhu, Wenqiao and Liu, Ji},
  journal={arXiv preprint arXiv:2505.19457},
  year={2025}
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* We would like to thank [Weijie Zhang](https://github.com/zhangwj618) for his contribution to the development of the inference engine.
* This work leverages [vLLM](https://github.com/vllm-project/vllm) as the backend model server for evaluation purposes.

