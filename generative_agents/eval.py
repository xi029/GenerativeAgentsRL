import requests
import argparse
import json
import os
import re
from typing import Dict, List, Optional
from openai import OpenAI

# ===================== 配置项（按需修改） =====================
DEFAULT_API_KEY = os.environ.get('DEEPSEEK_API_KEY') or "sk-e1fecac7367d455582948fdbc52cc4e4" # 你的DeepSeek API Key
DEFAULT_BASE_URL = "https://api.deepseek.com"  # DeepSeek API 地址
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek 模型

# 评估维度
EVAL_DIMENSIONS = [
    "任务完成度", 
    "协作效率", 
    "行为一致性", 
    "逻辑连贯性"
]

# ===================== 核心函数 =====================
def load_simulation_log(log_path: str) -> str:
    """加载simulation.md日志文件"""
    if not os.path.exists(log_path):
        print(f"Error: 日志文件不存在: {log_path}")
        return ""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 简单过滤，只保留主要内容，避免Token过长
            # 假设simulation.md主要包含时间线和Agent行为
            return content
    except Exception as e:
        print(f"加载日志失败：{e}")
        return ""

def build_eval_prompt(log_content: str, dimensions: List[str]) -> str:
    """构造大模型打分的Prompt"""
    # 截取日志以适应上下文窗口，优先保留最近的交互或均匀采样
    # 这里简单截取前 12000 个字符 (约 3-4k token)
    truncated_log = log_content[:12000]
    if len(log_content) > 12000:
        truncated_log += "\n...(日志截断)..."

    prompt = f"""
你是一位多智能体系统（Multi-Agent System）的专家评估员。请根据提供的仿真日志片段，对智能体的表现进行量化评估。

### 仿真日志片段
{truncated_log}

### 评估任务
请对上述仿真过程进行打分，评分范围 0-10 分（保留1位小数）。

### 评估维度
1. **任务完成度**：Agent 是否有效地推进了其既定目标（如写作、研究、社交等）？
2. **协作效率**：Agent 之间的交互（对话、等待）是否有效促进了信息共享或任务协作？是否存在无效的复读或死循环？
3. **行为一致性**：Agent 的行为是否符合其人设（如职业、性格）以及时间/空间逻辑？
4. **逻辑连贯性**：Agent 的行为序列（思考->计划->行动）是否逻辑自洽？

### 输出格式
请仅输出一个合法的 JSON 对象，不要包含 markdown 格式标记或其他废话。格式如下：
{{
    "任务完成度": 0.0,
    "协作效率": 0.0,
    "行为一致性": 0.0,
    "逻辑连贯性": 0.0,
    "简评": "一句话评价"
}}
"""
    return prompt.strip()

def call_llm_api(prompt: str, model: str, base_url: str, api_key: str) -> Dict:
    """调用 LLM API 获取打分结果"""
    
    try:
        print(f"正在请求模型 {model} 进行评估...")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.1, # 低温度以保证评估客观性
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # 尝试提取 JSON
        # 移除可能的 <think> 标签 (针对 DeepSeek-R1 等)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        
        # 提取 JSON 块
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            print(f"无法解析 JSON: {content}")
            return {}
            
    except Exception as e:
        print(f"API调用失败：{e}")
        return {}


def generate_eval_report(
    log_path: str, 
    label: str,
    model: str,
    base_url: str,
    api_key: str
) -> Dict:
    """生成单份日志的评估报告"""
    print(f"\n===== 开始评估: {label} =====")
    log_content = load_simulation_log(log_path)
    if not log_content:
        return {}
    
    prompt = build_eval_prompt(log_content, EVAL_DIMENSIONS)
    score_dict = call_llm_api(prompt, model, base_url, api_key)
    
    if not score_dict:
        print("评估失败，未获取到有效分数。")
        return {dim: 0.0 for dim in EVAL_DIMENSIONS}

    # 补全可能缺失的字段
    for dim in EVAL_DIMENSIONS:
        if dim not in score_dict:
            score_dict[dim] = 0.0
            
    # 计算总分
    valid_scores = [score_dict[d] for d in EVAL_DIMENSIONS]
    score_dict["总分"] = round(sum(valid_scores) / len(valid_scores), 1)
    
    print(f"[{label}] 评估结果：")
    for k, v in score_dict.items():
        print(f"  {k}: {v}")
        
    return score_dict

def main():
    parser = argparse.ArgumentParser(description="评估 Generative Agents 仿真结果")
    parser.add_argument("--before", help="优化前（Baseline）的 simulation.md 路径")
    parser.add_argument("--after", required=True, help="优化后（RL）的 simulation.md 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="评估使用的模型名称")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="API Base URL")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API Key")
    
    args = parser.parse_args()

    scores_before = {}
    if args.before:
        scores_before = generate_eval_report(args.before, "优化前(Baseline)", args.model, args.base_url, args.api_key)
    
    scores_after = generate_eval_report(args.after, "优化后(RL)", args.model, args.base_url, args.api_key)

    if scores_before and scores_after:
        print("\n===== ⚔️ 对比评估报告 ⚔️ =====")
        print(f"{'维度':<10} | {'优化前':<8} | {'优化后':<8} | {'变化':<8}")
        print("-" * 46)
        
        all_dims = EVAL_DIMENSIONS + ["总分"]
        for dim in all_dims:
            s1 = scores_before.get(dim, 0.0)
            s2 = scores_after.get(dim, 0.0)
            diff = s2 - s1
            diff_str = f"{diff:+.1f}"
            if diff > 0:
                trend = "🔺"
            elif diff < 0:
                trend = "🔻"
            else:
                trend = "➖"
                
            print(f"{dim:<10} | {s1:<8} | {s2:<8} | {trend} {diff_str}")

if __name__ == "__main__":
    main()
