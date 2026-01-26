import os
import psutil
import time
import re
import signal
import subprocess


def terminate_subprocess(proc: subprocess.Popen | psutil.Process):
    try:
        for s in [signal.SIGINT, signal.SIGTERM]:
            for _ in range(2):
                try:
                    children = psutil.Process(proc.pid).children()
                except psutil.NoSuchProcess:
                    pass
                else:
                    for child in children:
                        terminate_subprocess(child)
                try:
                    proc.send_signal(s)
                    proc.wait(timeout=3)
                    return
                except subprocess.TimeoutExpired:
                    continue
                except psutil.NoSuchProcess:
                    return
        proc.kill()
    except:
        pass


def print_subprocess_out_err(proc):
    outs, errs = proc.communicate()
    if outs:
        print(outs.decode())
    if errs:
        print(errs.decode())


def is_cpu_idle(procs: list[subprocess.Popen | psutil.Process], threshold=10.):
    """指定进程及所有子进程的CPU使用率低于阈值，则返回True"""
    ps = []
    for p in procs:
        try:
            ps.append(p := psutil.Process(p.pid))
            ps.extend(p.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    if not ps:
        return True

    # 检查3次，间隔0.5秒
    running_status = ['running', 'disk-sleep']
    for _ in range(3):
        for p in ps:
            try:
                if p.status() in running_status or p.cpu_percent(interval=0.1) > threshold:
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.5)
    return True


def is_gpu_idle():
    """所有GPU使用率为0，则返回True"""
    gpu_info = os.popen('nvidia-smi | grep MiB').read().split('\n')
    n_gpu = 0
    n_idle = 0
    for s in gpu_info:
        m = re.search(r'[0-9]+%', s)
        if m:
            n_gpu += 1
            if m.group(0) == '0%':
                n_idle += 1
    return n_gpu == n_idle
