import os
import sys
import argparse
import time
import datetime
import re
import json
import socket
import requests
import subprocess

from subprocess import Popen

from backend.utils import terminate_subprocess, print_subprocess_out_err, is_cpu_idle, is_gpu_idle

cur_path = os.path.dirname(os.path.abspath(__file__))

ngpus = int(subprocess.run(  # all available gpus
    'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l', shell=True, stdout=subprocess.PIPE
).stdout)


cur_args : list[str] = None  # 运行中的参数
pending_args : list[str] = None  # 需要运行的参数
proc_data_server : Popen = None
session = requests.Session()
multi_node_group = {}  # 多节点模型并行的ranks和head_ip等信息


def get_master_ip_addr(retry=10, retry_interval=10):
    master_addr = os.getenv('MASTER_ADDR')
    if master_addr is None or os.getenv('WORLD_SIZE', '1') == '1':
        return '127.0.0.1'
    for _ in range(retry):
        try:
            return socket.gethostbyname(master_addr)  # 将k8s域名解析为ip
        except socket.gaierror:  # 解析失败重试
            time.sleep(retry_interval)


def find_checkpoint(model_path, load_type):
    log_file = os.path.join(model_path, 'logging.jsonl')
    if not os.path.isfile(log_file):
        checkpoint = find_latest_checkpoint(model_path)

    else:
        with open(log_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        model_path_info = json.loads(lines[-1])
        last_model_checkpoint = model_path_info["last_model_checkpoint"]
        best_model_checkpoint = model_path_info["best_model_checkpoint"]
        if not best_model_checkpoint:
            best_model_checkpoint = last_model_checkpoint

        if load_type == "last":
            checkpoint = last_model_checkpoint
        if load_type == "best":
            checkpoint = best_model_checkpoint

    if os.path.isdir(f'{checkpoint}-merged'):  # LoRA合并后的目录
        checkpoint = f'{checkpoint}-merged'

    return checkpoint


def find_latest_checkpoint(model_path):
    cs = os.listdir(model_path)
    if not any(re.match(r'pytorch_model.*\.bin', f) or re.match(r'model.*\.safetensors', f) for f in cs):
        c2s = {}
        for c in cs:
            m = re.match(r'checkpoint-([0-9]+)$', c)
            if m:
                c2s[c] = int(m.group(1))
        if c2s:
            model_path = os.path.join(model_path, max(c2s, key=c2s.get))
            print(f'自动使用模型最新checkpoint：{model_path}')
    return model_path


def send_request(
    method, url,
    data=None, json=False, retry_interval=10, max_retries=10, ignore_errors=False
):
    """用于请求推理引擎，包括：提交文件、请求推理进度、发送终止信号"""
    assert method in ['get', 'post']
    retry_times = 0
    while True:
        try:
            if method == 'post':
                r = session.post(url, **{'json' if json else 'data': data})
            else:
                r = session.get(url, params=data)
            return r
        except (OSError, requests.exceptions.ConnectionError):
            if retry_times >= max_retries:
                if ignore_errors:
                    break
                else:
                    print(f'请求失败：重试超过{max_retries}次！')
                    raise
            time.sleep(retry_interval)
            retry_times += 1
            continue


def run_predict_remote(args):
    """
    调用远程inference服务
    """
    # 将数据提交至推理队列
    generation_params = {k: getattr(args, k) for k in [
        'max_new_tokens', 'temperature', 'top_p', 'stop', 'repetition_penalty', 'presence_penalty'
    ]}
    generation_params['n'] = args.sample_n
    data = {
        'input_path': args.data,
        'preprocess': args.preprocess,
        'postprocess': args.postprocess,
        'output_key': args.output_key,
        'overwrite': args.overwrite,
        'chat_template_kwargs': args.chat_template_kwargs,
        'generation_params': json.dumps(generation_params)
    }

    for k in ['output_dir', 'output_path']:
        if v := getattr(args, k):
            data[k] = v
    print('POST', args.model + '/file', data)
    r = send_request('post', args.model + '/file', data).json()
    assert r['status_code'] == 0, f'post file err: {r}'
    input_paths = r['input_path']
    output_paths = r['output_path']
    t_start = t_log = time.time()
    try:
        while True:
            time.sleep(1)
            # 获取文件推理进度
            n_pred = 0
            n_done = 0
            for p0, p1 in zip(input_paths, output_paths):
                r = send_request('get', args.model + '/file', {'input_path': p0, 'output_path': p1})
                try:
                    r = r.json()
                except requests.exceptions.JSONDecodeError:
                    raise RuntimeError('Data server 异常')
                assert r['status_code'] == 0, f'get file err: {r}'
                n_pred += r['n_pred']
                if r['is_done']:
                    n_done += 1
            # 打印日志
            s_time = f'[{datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")}] '
            if time.time() - t_log >= args.log_interval:
                t_log = time.time()
                print(
                    s_time + f'Finished {n_pred} samples '
                    f'({n_pred / (time.time() - t_start):.1f} samples/s, '
                    f'{n_done} files ({len(input_paths)} files in total)'
                )
            if n_done == len(input_paths):
                print(s_time + '全部预测完成!')
                break
    except KeyboardInterrupt:  # 提前退出
        for p0, p1 in zip(input_paths, output_paths):
            send_request(
                'get', args.model + '/file', {'input_path': p0, 'output_path': p1, 'abort': '1'},
                ignore_errors=True
            )


def run_predict_multi_gpu(args, engine_args: list[str]):
    """
    用于多卡/多机推理
    """
    env = os.environ.copy()
    env.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if 'NCCL_SOCKET_IFNAME' in env:
        env.setdefault('GLOO_SOCKET_IFNAME', env['NCCL_SOCKET_IFNAME'])

    args.server_addr = get_master_ip_addr()
    if args.device:
        gpus = args.device.split(',')
    else:
        gpus = list(map(str, range(ngpus)))
        args.device = ','.join(gpus)

    n_parallel = args.tensor_parallel * args.pipeline_parallel
    world_size = int(env.get('WORLD_SIZE', '1'))
    rank = os.getenv('RANK', 'unknown')
    ranks = args.rank.split(',') if args.rank else [str(i) for i in range(world_size)]
    is_node_0 = world_size == 1 or rank == '0'
    is_multi_node = world_size > 1 and n_parallel > len(gpus) and not args.is_idle  # 多节点模型并行
    use_ray = is_multi_node and args.backend == 'vllm'

    if is_node_0:
        global proc_data_server
        proc_data_server = start_data_server(args)

    if not args.is_idle:
        assert len(gpus) * len(ranks) >= n_parallel, 'GPU数量不足，请确认GPU数量大于或等于 tensor_parallel * pipeline_parallel 数值！'
        assert len(gpus) * len(ranks) % n_parallel == 0, 'GPU数量错误，请确认GPU数量能被 tensor_parallel * pipeline_parallel 整除！'
        send_request(
            'post', f'http://{args.server_addr}:{args.server_port}/node',
            {'rank': rank, 'args': cur_args}, json=True
        )

    # ------------------------- 启动 ray cluster（多机）-----------------------------
    if is_multi_node:
        n_nodes_per_model = n_parallel // len(gpus)
        head_ip, local_ranks = None, None  # 如：4节点推理，tp=16，local_ranks为[0, 1]或[2, 3]
        for i in range(0, len(ranks), n_nodes_per_model):
            if rank in ranks[i: i + n_nodes_per_model]:
                local_ranks = ranks[i: i + n_nodes_per_model]
                break
        assert local_ranks is not None, f'{rank=} {ranks=} {world_size=}'
        print(f'multi-node model parallel: {rank=} {n_nodes_per_model=} {local_ranks=}')
        while head_ip is None:
            r = send_request('get', f'http://{args.server_addr}:{args.server_port}/info')
            head_ip = r.json()['node_info'].get('ranks', {}).get(local_ranks[0], {}).get('ip')
        multi_node_group.update(head_ip=head_ip, local_ranks=local_ranks)

    start_worker = True
    if use_ray:
        if rank == local_ranks[0]:
            cmd = ['ray', 'start', '--head', '--port', env['MASTER_PORT']]
        else:
            cmd = ['ray', 'start', '--address', '{}:{}'.format(head_ip, env['MASTER_PORT'])]
            start_worker = False
        cmd.extend(['--min-worker-port', '1024', '--max-worker-port', '1200'])  # 默认10002-19999，多机任务可能会导致端口冲突
        print(' '.join(cmd))
        subprocess.run(cmd, env=env)

    # ------------------------------ 开始推理 -------------------------------------
    try:
        if is_node_0:
            run_predict_master_node(args, engine_args, env)
        else:
            run_predict_worker_node(args, engine_args, env, start_worker)
    except:
        raise
    finally:
        if use_ray:
            subprocess.run(['ray', 'stop'])


def start_data_server(args):
    """
    启动 data server（负责数据读写）
    """
    cmd_data_server = [
        'python', os.path.join(cur_path, 'data_server.py'),
        '--preprocess', args.preprocess,
        '--prompt', args.prompt,
        '--output_key', args.output_key,
        '--output_type', args.output_type,
        '--port', str(args.server_port),
    ]
    if args.model and args.backend != "asr":
        cmd_data_server.extend([
            '--tokenizer', args.model,
            '--model_name', args.model_name,
        ])
    if args.data:
        cmd_data_server.extend(['--data', args.data])
    for k in [
        'output_dir', 'output_path', 'postprocess', 'max_length', 'stop', 'subsample', 'seed', 'file_schedule',
        'num_outputs_per_model', 'chat_template', 'chat_template_kwargs', 'tool_call_parser', 'reasoning_parser'
    ]:
        if v := getattr(args, k):
            cmd_data_server.extend([f'--{k}', str(v)])
    for k in ['reuse', 'overwrite', 'no_order', 'output_special_tokens']:
        if v := getattr(args, k):
            cmd_data_server.append(f'--{k}')
    if proc_data_server is None:
        print('Starting:', ' '.join(cmd_data_server))
        return Popen(cmd_data_server)
    else:
        print('Reloading data server:', ' '.join(cmd_data_server[2:]))
        send_request(
            'post', f'http://{args.server_addr}:{args.server_port}/data_args',
            {'args': cmd_data_server[2:]}, json=True
        )
        return proc_data_server


def start_predict_workers(args, engine_args, env, rank):
    """
    启动推理进程（多个）
    """
    gpus = args.device.split(',')
    proc_list = []
    cmd_predict_list = []
    for i in range(len(gpus) // min(len(gpus), args.tensor_parallel)) if not args.is_idle else []:
        device = gpus[i * args.tensor_parallel : (i + 1) * args.tensor_parallel]
        cmd_predict = [
            'python', os.path.join(cur_path, 'backend', f'predict_async_{args.backend}.py'),
            '--server_addr', args.server_addr,
            '--server_port', str(args.server_port),
        ]
        if args.model:
            cmd_predict.extend(['--model', args.model])
        cmd_predict.extend([
            '--output_type', args.output_type,
            '--max_new_tokens', str(args.max_new_tokens),
            '--temperature', str(args.temperature),
            '--top_p', str(args.top_p),
            '--repetition_penalty', str(args.repetition_penalty),
            '--presence_penalty', str(args.presence_penalty),
            '--sample_n', str(args.sample_n),
        ])
        if len(device) < ngpus:
            cmd_predict.extend(['--device', ','.join(device)])
        for k in [
            'max_length', 'max_time', 'stop'
        ]:
            if v := getattr(args, k):
                cmd_predict.extend([f'--{k}', str(v)])
        if args.low_vram:
            cmd_predict.append('--low_vram')
        if args.tensor_parallel > 1:
            cmd_predict.extend(['--tensor_parallel', str(args.tensor_parallel)])
        if args.pipeline_parallel > 1:
            cmd_predict.extend(['--pipeline_parallel', str(args.pipeline_parallel)])
        if args.backend == 'sglang' and multi_node_group:
            cmd_predict.extend(['--head_ip', multi_node_group['head_ip']])
            cmd_predict.extend(['--local_rank', str(multi_node_group['local_ranks'].index(rank))])
        cmd_predict.extend(engine_args)
        cmd_predict_list.append(cmd_predict)

    send_request(
        'post', f'http://{args.server_addr}:{args.server_port}/node',
        {'rank': rank, 'n_workers': len(cmd_predict_list)}, json=True
    )

    if args.manual_start:
        for cmd in cmd_predict_list:
            print('Manual start:', ' '.join(cmd))
    elif cmd_predict_list:  # 启动第一个worker（其余worker延迟启动，防止多个vllm同时访问torch_compile_cache，导致异常）
        start_cmd_list([cmd_predict_list.pop(0)], env, proc_list)

    return proc_list, cmd_predict_list


def start_cmd_list(cmd_list: list[str], env: dict[str, str], proc_list: list[Popen]):
    """执行多条命令，用于启动推理子进程"""
    for cmd in cmd_list:
        print('Starting:', ' '.join(cmd))
        proc_list.append(Popen(cmd, env=env))
    cmd_list.clear()


def update_pending_args(a: list[str]):
    """检测启动参数是否有更新"""
    if a and ' '.join(a) != ' '.join(cur_args):
        global pending_args
        pending_args = a
        print('即将重启推理引擎，使用参数:', pending_args)
        return True


def run_predict_master_node(args, engine_args, env):
    """
    用于单机，或多机的头（master）节点
    基于vllm，启动多个推理进程（backend），并通过统一的data_server实现数据的异步读取和写入，以及推理进程之间的负载均衡。
    """
    # ------------------------ 启动 data server 和推理子进程---------------------------
    rank =  os.getenv('RANK', 'unknown')
    proc_predict_list, cmd_predict_list = start_predict_workers(args, engine_args, env, rank)
    proc_list = [proc_data_server] + proc_predict_list

    # -------------------------- 推理过程中，打印信息 -----------------------------------
    n_pred_0 = 0
    t0 = t_log = time.time()
    err = ''
    try:
        while True:
            time.sleep(1)
            r = send_request(
                'get', f'http://{args.server_addr}:{args.server_port}/info', {'rank': rank},
                max_retries=0, ignore_errors=True
            )
            if r is not None:
                try:
                    r = r.json()
                except requests.exceptions.JSONDecodeError:
                    terminate_subprocess(proc_data_server)
                    print_subprocess_out_err(proc_data_server)
                    err = 'Data server 异常'
                    break
                samples_per_sec = r['n_pred'] / (r['t_curr'] - r['t_start'])
                tokens_per_sec = r['n_token'] / (r['t_curr'] - r['t_start'])
                s_time = f'[{datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")}] '
                if not args.run_forever and not r['n_pred']:
                    if time.time() - t_log >= args.log_interval:
                        t_log = time.time()
                        print(s_time + 'loading...')
                    if time.time() - t0 > 3600 and is_cpu_idle(proc_list):
                        print('60分钟没有加载完成，超时退出...')
                        for proc in proc_list[1:]:
                            terminate_subprocess(proc)
                            print_subprocess_out_err(proc)
                        err = '60分钟没有加载完成，超时退出'
                        break
                if update_pending_args(r.get('global_args')):  # 下游任务发送了新的启动参数
                    break
                if cmd_predict_list and not r['n_pred']:
                    pass  # 推理进程可能还没启动，不打印日志
                else:
                    s_skipped = f', skipped {r["n_skip"]} samples >= max len' if r["n_skip"] else ''
                    if time.time() - t_log >= args.log_interval:
                        t_log = time.time()
                        print(
                            s_time + f'Finished {r["n_pred"]} samples '
                            f'({samples_per_sec:.1f} samples/s, {tokens_per_sec:.1f} tokens/s{s_skipped}), '
                            f'{r["n_done"]} files ({r["n_file"]} files in total)'
                        )
                    if not args.run_forever and n_pred_0 and r['n_pred'] == n_pred_0:
                        if time.time() - t0 > 1800 and is_gpu_idle():  # 30分钟没有预测完成的样本，推理进程卡住不动？-> 检查GPU使用率
                            for proc in proc_list[1:]:
                                terminate_subprocess(proc)
                                print_subprocess_out_err(proc)
                            err = '超过30分钟没有预测完成的样本，且GPU使用率为0，超时退出'
                            break
                    elif r['n_pred']:
                        n_pred_0 = r['n_pred']
                        t0 = time.time()
                if (
                    not args.manual_start and cmd_predict_list  # 当前节点有多个worker，第一个worker启动中，其余worker在等待
                    and 'n_queue' in r['node_info'].get('ranks', {}).get(rank, {})  # 第一个worker已启动完成，启动其余worker
                ):
                    start_cmd_list(cmd_predict_list, env, proc_list)
                if r['post_start']:  # 下游发送启动信号（当manual_start为True时）
                    start_cmd_list(cmd_predict_list, env, proc_list)
                if r['terminated']:  # 下游任务已发送结束信号
                    break
                if r.get('error'):
                    terminate_subprocess(proc_data_server)
                    print_subprocess_out_err(proc_data_server)
                    err = 'Data server 异常'
                    break
                if not args.run_forever and r['n_done'] == r['n_file']:  # 全部预测完成
                    print(s_time + '全部预测完成!')
                    send_request(
                        'post', f'http://{args.server_addr}:{args.server_port}/terminate',
                        max_retries=1, ignore_errors=True
                    )
                    break
                for i, proc in enumerate(proc_list[1:]):  # 推理进程异常退出
                    if proc.poll() is not None:  # poll() -> returncode
                        print_subprocess_out_err(proc)
                        err = f'推理进程{i}异常退出'
                if err:
                    break

            if proc_data_server.poll() is not None:  # terminated
                print_subprocess_out_err(proc_data_server)
                err = 'Data server 异常退出'
                break

    except KeyboardInterrupt:  # 提前退出
        pass

    # ---------------------------- 预测完成，结束子进程 ---------------------------------
    time.sleep(5)  # 等待子进程自行退出
    for proc in (proc_list[1:] if pending_args else proc_list):  # 如果有待启动的参数，不中止data server
        terminate_subprocess(proc)

    if err:
        raise RuntimeError(err)


def run_predict_worker_node(args, engine_args, env, start_worker):
    """
    用于多机worker节点
    """
    rank =  os.getenv('RANK', 'unknown')
    if start_worker:
        proc_predict_list, cmd_predict_list = start_predict_workers(args, engine_args, env, rank)
    else:
        proc_predict_list, cmd_predict_list = [], []

    check_interval = 3
    time.sleep(120)  # 留多一点时间等master节点启动
    try:
        while True:
            time.sleep(check_interval)
            r = send_request(
                'get', f'http://{args.server_addr}:{args.server_port}/info', {'rank': rank},
                max_retries=10, ignore_errors=True
            )
            if r is None:
                break
            else:
                try:
                    r = r.json()
                except requests.exceptions.JSONDecodeError:
                    break
                if update_pending_args(r.get('global_args')):  # 下游任务发送了新的启动参数
                    break
                if (
                    not args.manual_start and cmd_predict_list  # 当前节点有多个worker，第一个worker启动中，其余worker在等待
                    and 'n_queue' in r['node_info'].get('ranks', {}).get(rank, {})  # 第一个worker已启动完成，启动其余worker
                ):
                    start_cmd_list(cmd_predict_list, env, proc_predict_list)
                if r['post_start']:  # 下游发送启动信号（当manual_start为True时）
                    start_cmd_list(cmd_predict_list, env, proc_predict_list)
                if r['terminated']:  # 下游任务已发送结束信号
                    break
                if not args.run_forever and r['n_done'] == r['n_file']:  # 全部预测完成
                    break
                if r.get('error'):
                    break
                for proc in proc_predict_list:  # 推理进程异常退出
                    if proc.poll() is not None:  # poll() -> returncode
                        print_subprocess_out_err(proc)
                        break
    except KeyboardInterrupt:  # 提前退出
        pass

    time.sleep(5)  # 等待子进程自行退出
    for proc in proc_predict_list:
        terminate_subprocess(proc)


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--data", type=str, help="Data file to predict")
    parser.add_argument("--preprocess", type=str, default='preprocess', help="Module that provides preprocessing function")
    parser.add_argument("--postprocess", type=str, default=None, help="Module that provides postprocessing function")
    parser.add_argument("--prompt", type=str, default='Hithink', help="Prompt type")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--output_key", type=str, default='predict_result', help="Output key")
    parser.add_argument("--output_type", type=str, default='text', help="Output type")
    parser.add_argument("--model_name", type=str, default=None, help="Used by some postprocess methods, e.g. postprocess_append_choices.py")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--max_input_tokens", type=int, default=None,
                        help="Max number of input tokens, longer inputs will be truncated")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_time", type=int, default=None, help="Max number of seconds per sample")
    parser.add_argument("--temperature", type=float, default=0., help="0.0 means greedy decoding")
    parser.add_argument("--top_p", type=float, default=1., help="1.0 means sampling from all tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="1.0 means no penalty")
    parser.add_argument("--presence_penalty", type=float, default=0., help="0.0 means no penalty")
    parser.add_argument("--sample_n", "--beam_size", type=int, default=1, help="generate n outputs by sampling, or beam_size when output_type=beam search")
    parser.add_argument("--stop", type=str, default=None, help="token/word at which generation will be stopped")
    parser.add_argument("--reuse", action='store_true',
                        help="If prediction file exists, reuse previous outputs and only predict samples with empty output")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite if output file exists. If not set, will skip finished samples")
    parser.add_argument("--no_order", action='store_true', help="Output will be written in different order than input")
    parser.add_argument("--server_addr", type=str, help="Data server address")
    parser.add_argument("--server_port", type=int, default=1788, help="Data server port")
    parser.add_argument("--backend", type=str, default='vllm', choices=['vllm', 'hf', 'sglang', 'flag_embedding', 'faiss', 'tgi','asr'], help="Inference backend")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Number of tensor parallel gpus")
    parser.add_argument("--pipeline_parallel", type=int, default=1, help="Number of pipeline stages")
    parser.add_argument("--device", type=str, default=None, help="e.g. 0,1,2... If not specified, use all gpus by default")
    parser.add_argument("--rank", type=str, default=None, help="e.g. 0,1,2... If not specified, use all nodes by default")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    parser.add_argument("--log_interval", type=float, default=5., help="seconds between printed logs")
    parser.add_argument("--run_forever", action='store_true', help="If not set, program will quit when there are no data to predict")
    parser.add_argument("--manual_start", action='store_true', help="If set, workers (backend) will not start automatically")
    parser.add_argument("--is_idle", action='store_true', help="If set, workers (backend) will not start on the current node")
    parser.add_argument("--load_type", type=str, default='last', choices=['last', 'best'], help="Load the latest model or the best performing model on the validation set")
    parser.add_argument("--subsample", type=float, help="proportion (0.0 - 1.0) or number (>= 1) of samples used")
    parser.add_argument("--seed", type=int, help="seed used for subsampling")
    parser.add_argument("--file_schedule", type=str, help="Schedule strategy for files")
    parser.add_argument("--num_outputs_per_model", type=int, help="limit the number of outputs by model_name (by counting the existing answers in 'choices')")
    parser.add_argument("--chat_template", type=str, help="chat template (text or .jinja file)")
    parser.add_argument("--chat_template_kwargs", type=str, help="kwargs for apply_chat_template, e.g. enable_thinking=False")
    parser.add_argument("--output_special_tokens", action='store_true', help="If set, use skip_special_tokens=False for tokernize.decode")
    parser.add_argument("--tool_call_parser", type=str, help="Used to parse the model-generated tool call into OpenAI API")
    parser.add_argument("--reasoning_parser", type=str, help="Used to parse the model-generated reasoning content into OpenAI API")
    args, engine_args = parser.parse_known_args(cur_args)  # 其余入参默认传给推理引擎

    is_remote = args.model and args.model.startswith('http://')
    if args.model and not is_remote:
        args.model = find_checkpoint(args.model, args.load_type)
        print(f"model path is {args.model}...")
        if not args.model_name:
            if 'runs' in args.model or 'checkpoint' in args.model:  # 训练过的模型，默认使用完整路径
                args.model_name = args.model
            else:  # 开源模型，默认使用模型目录名称
                args.model_name = os.path.basename(args.model)

    if args.data:
        assert args.output_dir or args.output_path, '未指定输出路径！（--output_dir 或 --output_path）'
        if os.path.isdir(args.data):
            assert args.output_dir, '输入（--data）是目录，需要指定输出路径！（--output_dir ）'
        if args.output_dir:
            assert args.output_dir != args.data, '输入路径（--data）与输出路径（--output_dir）不能相同！'

    if args.prompt == 'llama3':
        args.stop = ','.join([args.stop, '<|eot_id|>']) if args.stop else '<|eot_id|>'

    if not args.is_idle and args.rank:
        args.is_idle = (rank := os.getenv('RANK')) and rank not in args.rank.split(',')  # 不在指定的rank（node）中，则不启动推理进程

    if is_remote:
        run_predict_remote(args)
    else:
        if args.output_type == 'cosyvoice2':
            print('[INFO] preparing env for cosyvoice2')
            from utils.cosyvoice import prepare_cosyvoice_env
            prepare_cosyvoice_env(args.model)
        run_predict_multi_gpu(args, engine_args)


if __name__ == "__main__":
    pending_args = sys.argv[1:]
    while pending_args:
        cur_args = pending_args
        pending_args = None
        run()
