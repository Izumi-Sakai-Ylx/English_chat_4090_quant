## 1. 启动 vllm 服务
在终端1中启动 vllm ,以 int8 为例  
'''
python -m vllm.entrypoints.openai.api_server \
  --model /nas-xinchen/private/yulixuan/Throughput_Performance/Mistral/Mistral-12B-GPTQ-int8 \
  --quantization gptq \
  --port 28000 \
  --host 0.0.0.0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --served-model-name Mistral-12B-INT8
'''  
当出现每隔 10s 一次的日志输出时，说明服务已经启动成功

## 2. 测试服务
在 终端2 中运行以下命令来测试服务：  
'''
python test_int8.py
'''  
注：这是一个英文模型，所以用中文的 case 去测试他没什么意义，我们要用英文的 case 去测试他

## 3. 观察输出是否正常
正常后就可以开始测速
