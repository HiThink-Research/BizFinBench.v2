
export API_NAME=chatgpt # The api name, current support chatgpt
export API_KEY=xxx # Your api key
export MODEL_NAME=gpt-4.1

# Pass in the config file path to start evaluation
cd ..
python run.py --config config/offical/BizFinBench_v2.yaml --model_name ${MODEL_NAME}