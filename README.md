<p align="center">
  <h1 align="center">
    <img src="static/logo.png" alt="BizFinBench logo" height="40" style="position:relative; top:6px;">
  BizFinBench.v2: A Unified Offline–Online Bilingual Benchmark for Expert-Level Financial Capability Evaluation of LLMs</h1>
    <p align="center">
    <span class="author-block">
      <!-- Xin Guo<sup>1,2,*</sup> </a>,</span>-->
                <span class="author-block">
      <!--Rongjunchen Zhang<sup>1,*,♠</sup></a>, Guilong Lu<sup>1</sup>, Xuntao Guo<sup>1</sup>, Jia Shuai<sup>1</sup>, Zhi Yang<sup>2</sup>, Liwen Zhang<sup>2,♠</sup>-->
    </span>
    </div>
    <div class="is-size-5 publication-authors" style="margin-top: 10px;">
        <span class="author-block">
            <!--<sup>1</sup>HiThink Research, <sup>2</sup>Shanghai University of Finance and Economics -->
        </span>
        <br>
        <span class="author-block">
            <!--<sup>*</sup>Co-first authors, <sup>♠</sup>Corresponding author, zhangrongjunchen@myhexin.com,zhang.liwen@shufe.edu.cn -->
        </span>
    </div>
  </p>
  <p>
  <!--📖<a href="">Paper</a> |🏠<a href="">Homepage</a></h3>|🤗<a href="">Huggingface</a></h3>-->
  </p>
<div align="center"></div>
<p align="center">

<!--**BizFinBench.v2** is the secend release of [BizFinBench](https://github.com/HiThink-Research/BizFinBench). It is built entirely on real-world user queries from Chinese and U.S. equity markets. It bridges the gap between academic evaluation and actual financial operations.-->
**BizFinBench.v2** is the first large-scale evaluation benchmark for large language models that is based on authentic business data from the Chinese and U.S. stock markets and integrates online assessment. Addressing the limitations of existing benchmarks, such as distorted samples and narrow scenarios, BizFinBench.v2 constructs eight fundamental tasks and two online tasks across four core business scenarios by clustering real user queries from financial platforms, resulting in 29,578 expert-level question-answer pairs. Experimental results indicate that a significant performance gap remains between mainstream LLMs and financial experts in relevant tasks. Through error analysis, this benchmark identifies the specific capability deficiencies of current models, breaks through traditional limitations, and provides a precise foundation for evaluating the effectiveness of large-scale LLM deployment in the financial sector.

<img src="static/score_sequence.png" alt="Evaluation Result">

### 🌟 Key Features

* **Authentic & Real-Time:** 100% derived from real financial platform queries, integrating online assessment capabilities.
* **Expert-Level Difficulty:** A challenging dataset of **29,578 Q&A pairs** requiring professional financial reasoning.
* **Comprehensive Coverage:** Spans **4 core business scenarios**, 8 fundamental tasks, and 2 online tasks.

### 📊 Key Findings
* **High Difficulty:** Even **ChatGPT-5** achieves only 61.5% accuracy on main tasks, highlighting a significant gap vs. human experts.
* **Online Prowess:** **DeepSeek-R1** outperforms all other commercial LLMs in dynamic online tasks, achieving a total return of 13.46% with a maximum drawdown of -8%.

<!--## 📢 News 
- 🚀 [06/01/2026] TBD-->

## 📕 Data Distrubution
BizFinBench.v2 contains multiple subtasks, each focusing on a different financial understanding and reasoning ability, as follows:

### 📊 Detailed Statistics
| Scenarios | Tasks | Avg. Input Tokens | # Questions |
|:---|:---|---:|---:|
| **Business Information Provenance** | Anomaly Information Tracing | 8,679 | 4,000 |
| | Financial Multi-turn Perception | 10,361 | 3,741 |
| | Financial Data Description | 3,577 | 3,837 |
| **Financial Logic Reasoning** | Financial Quantitative Computation | 1,984 | 2,000 |
| | Event Logic Reasoning | 437 | 4,000 |
| | Counterfactual Inference | 2,267 | 2,000 |
| **Stakeholder Feature Perception** | User Sentiment Analysis | 3,326 | 4,000 |
| | Financial Report Analysis | 19,681 | 2,000 |
| **Real-time Market Discernment** | Stock Price Prediction | 5,510 | 4,000 |
| | Portfolio Asset Allocation | — | — |
| **Total** | | **—** | **29,578** |


<img src="static/distribution.png" alt="Data Distribution" width="60%">

## 📚 Example
Coming Soon


## 🛠️ Usage
### Install requirements
```sh
pip install -r requirements.txt
```
### Quick Start – Evaluate a Local Model

```sh
python run_pipeline.py \
    --config config/offical/BizFinBench_v2.yaml \
    --model_path /mnt/data/llm/models/chat/Qwen3-0.6B \
```

### Quick Start – Evaluate a Local Model and Score with a Judge Model

```sh
python run_pipeline.py \
  --config config/offical/BizFinBench_v2.yaml \
  --model_path /mnt/data/llm/models/chat/Qwen3-0.6B \
  --remote_model_port 16666 \
  --prompt_type chat_template \
  --tensor_parallel 1
  --judge_model_path /mnt/data/llm/models/base/Qwen2.5-8B-Instruct \
  --judge_model_port 16667 \
  --judge_tensor_parallel 1

```

### Quick Start – Evaluate external apis (e.g., chatgpt)

```sh
export API_NAME=chatgpt # The api name, currently support chatgpt
export API_KEY=xxx # Your api key
export MODEL_NAME=gpt-4.1

# Pass in the config file path to start evaluation
python run.py --config config/offical/BizFinBench_v2.yaml --model_name ${MODEL_NAME}
```
> **Note**: You can adjust the API’s queries-per-second limit by modifying the semaphore_limit setting in envs/constants.py. e.g., GPTClient(api_name=api_name,api_key=api_key,model_name=model_name,base_url='https://api.openai.com/v1/chat/completions', timeout=600, semaphore_limit=5)

## ✒️Citation

```
Coming Soon
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

## 💖 Acknowledgement
* TBD

