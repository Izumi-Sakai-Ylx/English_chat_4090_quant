import os
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

os.makedirs("/nas-xinchen/private/yulixuan/Throughput_Performance/Mistral", exist_ok=True)

# 指定路径和校准样本数量 
model_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/Mistral/Mistral-12B"
quant_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/Mistral/Mistral-12B-GPTQ-int4" 

os.makedirs(quant_path, exist_ok=True)

# 校准数据集的样本数量
n_samples = 128

# 准备校准数据集 (保持您原来的在线加载方式)
print(f"加载 WikiText 数据集，并准备 {n_samples} 个样本进行校准...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_samples = [
    d["text"] for d in dataset.shuffle(seed=42).select(range(n_samples * 2)) 
    if d["text"].strip()
][:n_samples]
print(f"成功准备 {len(text_samples)} 个有效的校准样本。")

# 加载 Tokenizer 
print("加载 Tokenizer 中...")
# 确保 trust_remote_code=True 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 定义量化配置 (使用从 auto_gptq 导入的 BaseQuantizeConfig)
print("定义BaseQuantizeConfig量化配置...")
quantize_config = BaseQuantizeConfig(
    bits=4,                  # 量化位数
    group_size=128,          # 分组大小
    damp_percent=0.01,       # 阻尼百分比
    desc_act=False,          # 描述激活
    static_groups=False,     # 静态分组
    sym=True,                # 对称量化
    true_sequential=True,    # 真正顺序量化
)

# 加载模型并执行量化
print("开始从原始模型进行加载和量化...")

# 使用AutoGPTQForCausalLM加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantize_config,
    trust_remote_code=True
)

# 关键修复：将文本样本转换为模型输入格式
print("正在将校准样本转换为模型输入格式...")
encoded_samples = []
for text in text_samples:
    tokens = tokenizer(text, return_tensors="pt")
    encoded_samples.append({
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask
    })

# 执行量化
print("正在执行量化操作...")
model.quantize(
    examples=encoded_samples,  # 使用处理后的样本
    batch_size=1,              # 根据内存调整batch大小
    use_triton=False,          # 禁用triton
)


# 保存量化模型
print("量化完成，正在保存模型...")
model.save_quantized(quant_path)

# 保存Tokenizer
print(f"保存 Tokenizer 至: {quant_path}...")
tokenizer.save_pretrained(quant_path)

print(f"\nAutoGPTQ 量化全部完成！模型已保存至: {quant_path}")