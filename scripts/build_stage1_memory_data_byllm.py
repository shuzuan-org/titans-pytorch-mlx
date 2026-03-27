#!/usr/bin/env python3
"""
Generate training data using a custom LLM endpoint.
Based on the provided configuration for GLM-4.7 via Minimax API.
"""

import asyncio
import argparse
import json
import logging
import os
import random
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp to run this script: pip install aiohttp")
    exit(1)

try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets to run this script: pip install datasets")
    exit(1)

# =======================================================
# LLM Remote API Configuration
# =======================================================
LLM_REMOTE_API_KEY = os.environ.get("LLM_REMOTE_API_KEY", "")
LLM_REMOTE_MODEL = os.environ.get("LLM_REMOTE_MODEL", "glm-4.7")
LLM_REMOTE_BASE_URL = os.environ.get(
    "LLM_REMOTE_BASE_URL",
    "https://mini.origintask.cn/v1/chat/completions",
)
LLM_REMOTE_TEMPERATURE = 0.1
LLM_REMOTE_TOP_P = 0.1
LLM_REMOTE_MAX_TOKENS = 8000
LLM_REMOTE_RATE_LIMIT_DELAY = 1.0 # 稍微调快点，Minimax API 一般能撑住
LLM_REMOTE_CONCURRENCY = 5
# =======================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个长文本记忆与状态追踪（State Tracking）训练集构建专家。
阅读用户提供的真实对话记录。从中找出一个【随着对话的进行，发生了明确状态改变、计划改变、观念或事实改变】的核心细节。
如果没有发生改变，请提取任何需要由多句话前后论述才敲定的核心事实。

请直接返回合法的 JSON 对象，不要含有任何其他多余文本或 Markdown 标记（无需 ```json 头），必须确保能用 json.loads 直接解析。JSON 格式如下：
{
  "history_chunks": ["第一句发言", "第二句发言", ...], // 把原始对话按次序切成多句字符串列表（尽量少于20句，截取发生状态变更的关键片段）
  "question_chunk": "一句针对上述最终变化或最终核心事实的提问。例如：用户最终决定预订什么时间的车票？",
  "answer": "最简短的最终状态答案"
}"""

async def call_llm(session: aiohttp.ClientSession, dialogue: str, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {LLM_REMOTE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_REMOTE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请对以下对话进行信息提取并生成指定的 JSON 格式数据：\n\n{dialogue}"}
            ],
            "temperature": LLM_REMOTE_TEMPERATURE,
            "top_p": LLM_REMOTE_TOP_P,
            "max_tokens": LLM_REMOTE_MAX_TOKENS
        }
        
        try:
            async with session.post(LLM_REMOTE_BASE_URL, headers=headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"API Error [{response.status}]: {text}")
                    return None
                    
                data = await response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Cleanup markdown formatting
                if content.startswith("```"):
                    lines = content.split('\n')
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines[-1].startswith("```"): lines = lines[:-1]
                    content = "\n".join(lines).strip()
                         
                result = json.loads(content)
                if all(k in result for k in ["history_chunks", "question_chunk", "answer"]):
                    return result
                return None
        except Exception as e:
            logger.error(f"LLM request exception/validation failed: {e}")
            return None
        finally:
            await asyncio.sleep(LLM_REMOTE_RATE_LIMIT_DELAY)

def load_local_dialogues(dataset_path: str) -> list[str]:
    """Fallback native JSON parser designed specifically for CrossWOZ and other custom massive JSON files"""
    dialogues_out = []
    path = Path(dataset_path)
    
    files_to_check = []
    if path.is_dir():
        unzipped_dir = path / "unzipped"
        if unzipped_dir.exists():
            files_to_check.extend(unzipped_dir.glob("*.json"))
        files_to_check.extend(path.glob("**/*.json"))
        # Prioritize train files
        train_files = [f for f in files_to_check if "train" in f.name.lower()]
        target_files = train_files if train_files else files_to_check
        if not target_files:
            raise FileNotFoundError(f"No JSON files found in {dataset_path} or unzipped cache.")
        target_file = target_files[0]
    else:
        target_file = path
        
    logger.info(f"Natively parsing local JSON file: {target_file}")
    with open(target_file, "r", encoding="utf-8") as f:
        if target_file.suffix == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            
    # Normalize structure (CrossWOZ is dict of dicts, others might be lists)
    items = data.values() if isinstance(data, dict) else data
    
    for item in items:
        # CrossWOZ format specifically uses {"messages": [{"content": "...", "role": "usr"}]}
        if isinstance(item, dict) and "messages" in item and isinstance(item["messages"], list):
            dlg_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in item["messages"] if isinstance(m, dict)])
            if dlg_text: dialogues_out.append(dlg_text)
        elif isinstance(item, dict):
            # Generic fallback
            if "dialogue" in item: dialogues_out.append(str(item["dialogue"]))
            elif "text" in item: dialogues_out.append(str(item["text"]))
            elif "conversation" in item: 
                val = item["conversation"]
                dialogues_out.append("\n".join([str(x) for x in val]) if isinstance(val, list) else str(val))
            else:
                dialogues_out.append(json.dumps(item, ensure_ascii=False))
        elif isinstance(item, str):
            dialogues_out.append(item)
            
    return dialogues_out

async def main():
    parser = argparse.ArgumentParser(description="Use Local/Remote LLM to generator Memory Tracking Dataset")
    parser.add_argument("-n", "--count", type=int, default=10, help="Number of successful instances to generate")
    parser.add_argument("-d", "--dataset", type=str, default="samsum", help="HF Dataset name OR path to local JSON/JSONL")
    parser.add_argument("-o", "--output", type=str, default="../data/generated/stage1_llm_synth/train.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    if not LLM_REMOTE_API_KEY:
        raise ValueError("LLM_REMOTE_API_KEY environment variable is required")
    
    out_file = Path(args.output).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading '{args.dataset}' dataset...")
    dialogues = []
    
    # Try native logic first if it looks like a local folder or file
    is_local = args.dataset.endswith(".json") or args.dataset.endswith(".jsonl") or os.path.exists(args.dataset)
    
    if is_local:
        try:
            # HuggingFace Datasets disk format?
            if os.path.isdir(args.dataset) and os.path.exists(os.path.join(args.dataset, "dataset_info.json")):
                from datasets import load_from_disk
                ds = load_from_disk(args.dataset)
                raw_dataset = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
                for item in raw_dataset:
                    if "dialogue" in item: dialogues.append(str(item["dialogue"]))
                    elif "text" in item: dialogues.append(str(item["text"]))
                    else: dialogues.append(json.dumps(item, ensure_ascii=False))
            else:
                # Raw JSON dictionary parsing (e.g. CrossWOZ)
                dialogues = load_local_dialogues(args.dataset)
        except Exception as e:
            logger.warning(f"Native local parsing failed: {e}. Falling back to HF modules...")
            is_local = False
            
    if not dialogues and not is_local:
        try:
            raw_dataset = load_dataset(args.dataset, split="train", trust_remote_code=True)
            for item in raw_dataset:
                if "dialogue" in item: dialogues.append(str(item["dialogue"]))
                elif "text" in item: dialogues.append(str(item["text"]))
                elif "content" in item: dialogues.append(str(item["content"]))
                elif "conversation" in item:
                    val = item["conversation"]
                    dialogues.append("\n".join([str(x) for x in val]) if isinstance(val, list) else str(val))
                else:
                    dialogues.append(json.dumps(item, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to load dataset via HuggingFace: {e}")
            return
            
    if not dialogues:
        logger.error("No valid dialogues could be extracted from the dataset!")
        return

    random.seed(42)
    random.shuffle(dialogues)
    
    # We load 3x the required count to cover failed JSON parses or invalid requests
    pool_size = min(len(dialogues), args.count * 3)
    target_dialogues = dialogues[:pool_size]
    
    logger.info(f"Starting generation loop for {args.count} samples.")
    logger.info(f"Concurrency: {LLM_REMOTE_CONCURRENCY}, Rate Limit Delay: {LLM_REMOTE_RATE_LIMIT_DELAY}s")
    
    semaphore = asyncio.Semaphore(LLM_REMOTE_CONCURRENCY)
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for dg in target_dialogues:
            tasks.append(call_llm(session, dg, semaphore))
            
        generated_count = 0
        
        with open(out_file, "w", encoding="utf-8") as f:
            if tqdm:
                iterable = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying remote LLM")
            else:
                iterable = asyncio.as_completed(tasks)
                
            for future in iterable:
                res = await future
                if res is not None:
                    # Give it a unified episode ID
                    res["episode_id"] = f"{args.dataset}_{generated_count:06d}"
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                    generated_count += 1
                    
                if generated_count >= args.count:
                    break
                    
    logger.info(f"Generation fully complete! Successfully generated {generated_count} structured items to {out_file}")

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
