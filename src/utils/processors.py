import json
import os
import time
from typing import Optional, List, Dict, Any
from ..core.llm_client import generate_response

def remove_until_parenthesis(s: str) -> str:
    """○で始まる文字列から括弧までの部分を除去する"""
    if s.strip().startswith("○"):
        index1 = s.find(")")
        index2 = s.find("）")
        if index1 == -1 and index2 == -1:
            return s
        index = min(i for i in [index1, index2] if i != -1)
        return s[index + 1:]
    return s

def append_to_jsonl(entry: Dict[str, Any], filename: str) -> None:
    """JSONLファイルにエントリを追記する"""
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

def parse_strategies(response: Optional[str]) -> List[str]:
    """レスポンスから戦略をパースする"""
    if not response:
        return []
    
    strategies = []
    if '<start_strategies>' in response:
        strategy_section = response.split('<start_strategies>')[1]
    else:
        strategy_section = response
    
    lines = strategy_section.split('\n')
    current_strategy = []
    for line in lines:
        if line.strip().startswith('##'):
            if current_strategy:
                strategies.append('\n'.join(current_strategy))
            current_strategy = [line.strip()]
        elif line.strip() and current_strategy:
            current_strategy.append(line.strip())
    
    if current_strategy:
        strategies.append('\n'.join(current_strategy))
    return strategies

def generate_task_specific_strategies(chunk: str, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """チャンクからタスク固有の学習戦略を生成する"""
    prompt = f"""トリビアコンテストの勉強をしています。この文書に含まれるすべての情報をカバーする質問のリストを作成してください。すべての質問を作成した後、各質問について、その種類の情報を記憶するのに役立つ一般的な学習戦略またはプロンプトを生成してください（特定の質問にあまり焦点を当てすぎないでください）。プロンプトは、最も効果的に情報を内面化するために、どのように練習や演習を行うべきかについて、詳細なガイドラインまたはステップバイステップを概説する必要があります。
すべての質問を出力し、その後<start_strategies>、そしてすべての戦略を出力してください。各戦略の前に##を付けてください。日本語で出力してください。
<document>
{chunk}
</document>"""
    return generate_response(prompt, think=think, max_tokens=max_tokens)

def generate_task_agnostic_strategies(chunk: str, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """チャンクからタスク非依存の学習戦略を生成する"""
    prompt = f"""Consider the following document. What are some strategies specific to this document that I can use to help me learn and remember all of the information contained? Use markdown and prefix each strategy with ##
<document>
{chunk}
</document>"""
    return generate_response(prompt, think=think, max_tokens=max_tokens)

def apply_strategy_to_chunk(strategy: str, chunk: str, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """戦略をチャンクに適用して結果を得る"""
    prompt = f"""以下の学習戦略があります：
{strategy}
この戦略を以下の文書に適用してください。日本語で出力してください。
<document>
{chunk}
</document>"""
    return generate_response(prompt, think=think, max_tokens=max_tokens)