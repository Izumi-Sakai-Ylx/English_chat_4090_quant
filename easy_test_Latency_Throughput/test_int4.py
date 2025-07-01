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

def test_and_evaluate_int4():
    """
    主函数：使用数据集对 INT4 模型进行性能和质量评测。
    """
    print("=== 开始使用数据集评测 INT4 量化模型 ===")
    
    # 1. 指定数据集路径并加载数据
    dataset_path = "/nas-xinchen/private/yulixuan/Throughput_Performance/llm_throughput_eval/datasets/english_chat.jsonl"
    evaluation_cases = load_evaluation_cases_from_jsonl(dataset_path)
    
    if not evaluation_cases:
        print("数据集中没有可用的评测用例，测试终止。")
        return

    # 2. (可选) 选择评测的用例数量，便于快速测试
    cases_to_test = evaluation_cases[:10]
    print(f"将使用数据集中的 {len(cases_to_test)} 条用例进行评测。")

    # 3. 初始化 OpenAI 客户端，请确保端口号正确
    client = OpenAI(
        base_url="http://localhost:28001/v1",  # INT4 服务的端口
        api_key="no-key-required"
    )
    
    # 4. 准备存储结果的结构
    results = {
        "quant_type": "INT4",
        "dataset": dataset_path,
        "evaluation_results": []
    }
    
    # 注意：请根据您分析数据集得出的结论，设置一个合理的 max_tokens 值。
    # 这样可以避免模型输出因长度限制而被截断，从而影响质量评估。
    max_tokens = 512 

    # 5. 循环遍历用例进行评测
    for i, case in enumerate(cases_to_test):
        prompt = case["prompt"]
        ground_truth = case["ground_truth_output"]
        
        print(f"\n评测 {i+1}/{len(cases_to_test)}...")
        print(f"Prompt: {prompt[:80]}...")
        
        try:
            # 调用 vLLM API
            response = client.completions.create(
                model="Mistral-12B-INT4", # 确保模型名与您的服务一致
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens
            )
            
            model_output = response.choices[0].text.strip()
            
            # 将评测结果存入列表
            result_entry = {
                "case_id": i + 1,
                "prompt": prompt,
                "model_output": model_output,
                "ground_truth_output": ground_truth
            }
            results["evaluation_results"].append(result_entry)
            
            print(f"模型输出: {model_output[:60]}...")
            print(f"标准答案: {ground_truth[:60]}...")

        except Exception as e:
            print(f"请求失败: {str(e)}")
            continue
            
    # 6. 将最终结果保存到JSON文件
    with open("int4_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n=== INT4 评测完成 ===")
    print("评测结果已保存到 int4_evaluation_results.json 文件中，请打开查看并进行人工对比。")

if __name__ == "__main__":
    test_and_evaluate_int4()
