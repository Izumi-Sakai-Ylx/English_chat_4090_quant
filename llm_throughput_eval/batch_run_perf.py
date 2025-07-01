import multiprocessing
from multiprocessing import Manager
from pathlib import Path
import requests
from evalscope.perf.main import run_perf_benchmark
import os
import signal
import subprocess
import time
import json
from loguru import logger

def check_vllm_health(port):
    """检查vLLM服务健康状态"""
    url = f"http://localhost:{port}/health"
    try:
        response = requests.get(url, timeout=5) # 稍微延长超时以应对慢启动
        return response.status_code
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return None

def check_model_ready(port, model_name):
    """使用 vLLM 的 /v1/models API 来检查模型是否加载完成"""
    url = f"http://localhost:{port}/v1/models"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            for model_info in models_data.get("data", []):
                if model_info.get("id") == model_name:
                    logger.info(f"Found model '{model_name}' in the server's model list. Readiness confirmed.")
                    return True
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Model readiness check via /v1/models endpoint failed: {str(e)}")
        return False

def start_vllm(cmd, port, log_file_object, model_name):
    """启动vLLM服务并等待模型完全加载"""
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=log_file_object, stderr=subprocess.STDOUT,
            env=env, start_new_session=True
        )
        logger.info(f"Starting vLLM with PID: {process.pid} on port {port}")
    except Exception as e:
        logger.error(f"Failed to start vLLM: {e}")
        return None
        
    for _ in range(40): # 延长健康检查等待时间
        if check_vllm_health(port) == 200:
            logger.info(f"vLLM service healthy on port {port}")
            break
        time.sleep(5)
    else:
        logger.error(f"vLLM health check failed after 200 seconds")
        if process and process.poll() is None: process.kill()
        return None
        
    for _ in range(120):
        if check_model_ready(port, model_name):
            logger.info(f"Model {model_name} loaded successfully!")
            return process
        logger.info(f"Waiting for model {model_name} to load...")
        time.sleep(10)
    else:
        logger.error(f"Model {model_name} failed to load within 20 minutes")
        if process and process.poll() is None: process.kill()
        return None

def close_vllm(process: subprocess.Popen):
    """优雅停止vLLM服务"""
    if process is None: return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=30)
        logger.info(f"vLLM service stopped gracefully")
    except (subprocess.TimeoutExpired, ProcessLookupError, Exception) as e:
        logger.warning(f"Could not gracefully stop vLLM (PID: {process.pid}), terminating. Reason: {e}")
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception as term_e:
            logger.error(f"Error during vLLM termination: {term_e}")

def run(model_path, num_parallel, port, model_size, cuda_device, quantization=None,
        enable_chunked_prefill=False, swanlab_api_key=None, project_name=None):
    """运行单次性能测试流程（稳定版）"""
    
    # ------------------
    #  稳定版核心配置
    # ------------------
    # 恢复到已知可以成功运行的服务器配置
    vllm_max_model_len = 8192
    vllm_gpu_util = 0.9

    # 客户端参数必须与服务器兼容
    # 您的 7500 prompt 长度要求与 4096 的服务器容量冲突，这里强制修正以保证运行
    client_max_prompt = 7500 # 必须小于 vllm_max_model_len
    client_max_tokens = 512
    
    # 自动验证，确保配置逻辑正确
    if client_max_prompt + client_max_tokens > vllm_max_model_len:
         raise ValueError(f"Configuration Error: max_prompt({client_max_prompt}) + max_tokens({client_max_tokens}) > max_len({vllm_max_model_len})")

    cmd = (
        f"CUDA_VISIBLE_DEVICES={cuda_device} vllm serve {model_path} "
        f"--host 0.0.0.0 --port {port} --tensor-parallel-size 1 "
        f"--gpu-memory-utilization {vllm_gpu_util} "
        f"--max-model-len {vllm_max_model_len} "
        f"--dtype float16 " 
        f"--enable-prefix-caching --disable-sliding-window "
        f"--uvicorn-log-level warning --disable-log-requests "
        f"--chat-template src/templates/joyland_english_new.jinja"
    )
    if quantization:
        if quantization == 'awq':
            # 对于AWQ模型，使用日志推荐的、性能更好的 awq_marlin
            cmd += " --quantization awq_marlin"
        else: # 对于 'gptq'
            cmd += f" --quantization {quantization}"
    if enable_chunked_prefill:
        cmd += " --enable-chunked-prefill"
    # 添加你的路径
    output_dir = Path(f"machine_4090_{model_size}_parallel_{num_parallel}_backend_vllm")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vllm_cmd.txt").write_text(cmd)
    vllm_log_path = output_dir / f"vllm_server_{port}.log"
    logger.info(f"vLLM command: {cmd}") 
    logger.info(f"vLLM logs will be written to {vllm_log_path}")

    process = None
    
    # Bug修复：将 experiment_name 的定义移到 try 块之前
    quant_str = quantization if quantization else "fp16"
    base_name = f"{model_size}_p{num_parallel}_q{quant_str}" # 先构建基础名称
    if enable_chunked_prefill:
        base_name += "_chunked"

    # 在基础名称前统一添加 "test_" 前缀
    experiment_name = f"test_{base_name}"

    try:
        with vllm_log_path.open("w") as f_log:
            process = start_vllm(cmd, port, f_log, model_path)
            
        if process is None:
            raise RuntimeError(f"Failed to start vLLM service for model {model_path}!")
        
        args = {
            "model": model_path,
            "url": f"http://localhost:{port}/v1/chat/completions",
            "number": 1500, "parallel": num_parallel,
            "dataset": "custom",
            # 填入你实际的数据集路径
            "dataset_path": "xxx",
            "tokenizer_path": model_path, "outputs_dir": str(output_dir),
            "name": experiment_name, "swanlab_api_key": swanlab_api_key,
            "max_tokens": client_max_tokens, "temperature": 0.7,
            "seed": 2025, "stream": True, "max_prompt_length": client_max_prompt
        }
        
        logger.info(f"Starting benchmark for experiment: {experiment_name}...")
        run_perf_benchmark(args)
        logger.success(f"Benchmark for experiment {experiment_name} completed.")

    except Exception as e:
        logger.error(f"An exception occurred during benchmark execution for {experiment_name}: {str(e)}")
        raise
    finally:
        if process:
            close_vllm(process)

def run_task(task_args, gpu_queue, swanlab_api_key, project_name):
    """任务执行函数"""
    logger.critical(f"########## RUN_TASK IS RUNNING IN PROCESS ID: {os.getpid()} ##########")
    model_path, model_size, num_parallel, quant, enable_chunked_prefill = task_args
    gpu_id = gpu_queue.get()
    try:
        port = 9853 + 10 * int(gpu_id.split(',')[0])
        logger.info(f"Starting task on GPU {gpu_id}, port {port}: {model_size} parallel={num_parallel}")
        run(model_path, num_parallel, port, model_size, gpu_id, quant,
            enable_chunked_prefill, swanlab_api_key, project_name)
    except Exception as e:
        logger.error(f"Task for {model_size} p={num_parallel} failed: {str(e)}")
    finally:
        gpu_queue.put(gpu_id)
        logger.info(f"Finished task on GPU {gpu_id}, returning it to the queue.")

def main():
    """主函数"""
    SWANLAB_API_KEY = "iHXkcABCOH2bTH0u0nAXl"
    PROJECT_NAME = "LLM_4090_Throughput_Eval_Mistral12B"

    model_configs = [
        # 填入你模型的路径和名字，以及量化类型
        ("model___path", "model__name", "gptq"),
    ]
    # parallel一般设置成[8 16 32 64]
    parallel_sizes = [8]

    tasks = []
    for model_path, model_size, quant in model_configs:
        for num_parallel in parallel_sizes:
            tasks.append((model_path, model_size, num_parallel, quant, True))

    manager = Manager()
    gpu_queue = manager.Queue()
    gpu_ids = ["0"]
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    num_processes = len(gpu_ids)
    logger.info(f"Starting process pool with {num_processes} workers")


    for task_config in tasks:
            logger.info(f"########## Preparing to start a new, isolated process for task: {task_config[1]} p={task_config[2]} ##########")
            
            # 1. 为当前任务创建一个全新的进程
            process = multiprocessing.Process(
                target=run_task, 
                args=(task_config, gpu_queue, SWANLAB_API_KEY, PROJECT_NAME)
            )
            
            # 2. 启动进程
            process.start()
            
            # 3. 等待该进程完全执行结束
            process.join()
            
            logger.info(f"########## Process for task {task_config[1]} p={task_config[2]} has terminated. ##########\n")
            # 短暂休眠，给系统一点时间回收资源，虽然不是必须，但有时能增加稳定性
            time.sleep(5) 

    logger.success("All tasks completed!")


if __name__ == "__main__":
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add("batch_run_perf_final.log", rotation="500 MB", retention="7 days", level="INFO", format=log_format)
    logger.info("Starting benchmark script...")
    main()