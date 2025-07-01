import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 配置路径
model_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/Mistral/Mistral-12B"
quant_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/Mistral/Mistral-12B-AWQ"

# 初始化量化器
quantizer = AutoAWQForCausalLM.from_pretrained(model_path)

# 量化配置（W4A16，分组大小128）
quant_config = {"w_bit": 4, "q_group_size": 128, "version": "GEMM"}

# 执行量化
quantizer.quantize(
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
    quant_config=quant_config,
    calib_data="mit-han-lab/pile-val-backup",  # 校准数据集
    split="validation",  # 使用验证集
    text_column="text"    # 数据集文本字段
)

# 保存量化模型
quantizer.save_quantized(quant_path)
print(f"量化模型已保存至：{quant_path}")