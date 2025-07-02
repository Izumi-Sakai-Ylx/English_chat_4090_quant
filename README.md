大模型吞吐性能测评

本项目将对 LLM 进行吞吐性能测评
所用模型为： Mistral-12B
所用的机器： 4090D

基本实验步骤：

0. 所需依赖和配置环境
见 requirements.txt

1. 模型的量化
分别对模型进行 AWQ-int4 量化  GPTQ-int4 量化 和 GPTQ-int8 量化
见 quant 

2. 量化模型的简单评测
需要先启动vllm服务
见 easy_test_Latency_Throughput

3. 进行评测

依赖: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html

数据集：English_chat_4090_quant.html/llm_throughput_eval/datasets/english_chat.jsonl

运行脚本
```
python batch_run_perf.py
```
脚本说明：
这个脚本是测试不同并发下，LLM 的吞吐性能等指标在不同并发下的表现；
该脚本会先启动一个 vllm 脚本，然后通过evalscope.perf 来调用 vllm 服务，从而进行吞吐性能测试。
需要注意保存的目录命名，最终绘制指标图的时候，会根据目录名称解析运行的一些参数。

4. 量化结果评估
结果显示，GPTQ4的效果最好

