from openai import OpenAI
import time
import numpy as np
import json

def load_evaluation_cases_from_jsonl(file_path):
    """
    从JSON Lines文件中加载评测用例。
    每个用例包含'prompt'和'ground_truth_output'。
    """
    evaluation_cases = []
    print(f"正在从 {file_path} 加载评测用例...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    messages = data.get('messages', [])
                    
                    # 找到第一个user-assistant对话对
                    for i, message in enumerate(messages):
                        if message.get('role') == 'user' and i + 1 < len(messages):
                            next_message = messages[i+1]
                            if next_message.get('role') == 'assistant':
                                prompt = message.get('content', '')
                                ground_truth = next_message.get('content', '')
                                
                                if prompt and ground_truth:
                                    evaluation_cases.append({
                                        "prompt": prompt,
                                        "ground_truth_output": ground_truth
                                    })
                                # 只提取第一个对话对
                                break
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"跳过格式错误的一行: {line.strip()} - 错误: {e}")
                    continue
        print(f"成功加载 {len(evaluation_cases)} 条评测用例。")
        return evaluation_cases
    except FileNotFoundError:
        print(f"错误: 数据集文件未找到于 {file_path}")
        return []

def test_and_evaluate_awq():
    """
    此函数用于连接到 vLLM 服务器，使用指定数据集评测 AWQ 量化模型，
    并将结果保存到 JSON 文件中以供分析。
    """
    print("=== 开始使用数据集评测 AWQ 量化模型 ===")
    
    # 使用与之前相同的测试数据集
    dataset_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/llm_throughput_eval/datasets/english_chat.jsonl"
    evaluation_cases = load_evaluation_cases_from_jsonl(dataset_path)
    
    if not evaluation_cases:
        print("数据集中没有可用的评测用例，测试终止。")
        return

    # 选择评测数量 (例如前10条)
    cases_to_test = evaluation_cases[:10]
    print(f"将使用数据集中的 {len(cases_to_test)} 条用例进行评测。")

    # 初始化 OpenAI 客户端，连接到本地 vLLM 服务器
    client = OpenAI(
        base_url="http://localhost:28002/v1", # <-- 注意：端口已更新
        api_key="no-key-required"
    )
    
    results = {"evaluation_results": []}
    
    for i, case in enumerate(cases_to_test):
        prompt = case["prompt"]
        ground_truth = case["ground_truth_output"]
        
        print(f"\n评测 {i+1}/{len(cases_to_test)}...")
        print(f"Prompt: {prompt[:80]}...")
        
        try:
            # 调用在服务器启动时定义的 AWQ 模型
            response = client.completions.create(
                model="Mistral-12B-AWQ",
                prompt=prompt,
                temperature=0,
                max_tokens=256 # 可根据需要调整
            )
            
            model_output = response.choices[0].text.strip()
            
            result_entry = {
                "case_id": i + 1,
                "prompt": prompt,
                "model_output": model_output,
                "ground_truth_output": ground_truth
            }
            results["evaluation_results"].append(result_entry)
            
            print(f"AWQ 输出: {model_output[:60]}...")
            print(f"标准输出: {ground_truth[:60]}...")

        except Exception as e:
            print(f"请求失败: {str(e)}")
            continue
            
    # 将结果保存到新的 JSON 文件中
    output_filename = "awq_evaluation_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n=== 评测完成 ===")
    print(f"评测结果已保存到 {output_filename} 文件中，请打开查看并进行人工对比。")

if __name__ == "__main__":
    test_and_evaluate_awq()
