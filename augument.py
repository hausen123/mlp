import os
import time
import asyncio
from src.utils.processors import (
    remove_until_parenthesis,
    append_to_jsonl,
    parse_strategies,
    generate_task_specific_strategies,
    apply_strategy_to_chunk
)
from src.core.llm_client import generate_response

def fact_distillation(chunk: str, think: bool = False, max_tokens: int = None) -> str:
    """チャンクからatomic factsを抽出する"""
    
    prompt = f"""あなたは原子力安全規制の専門知識を持つ辞書編集者です。原子力規制に関する専門辞書に掲載する独立した事実を抽出してください。

以下の文章から、辞書の項目として価値のある本質的な事実のみを日本語で抽出してください。

重要な抽出基準：
1. 独立性：各事実は他の文脈に依存せず、単独で理解できること
2. 本質性：年月日、組織名、法律名、事故名など具体的で検証可能な情報を含むこと
3. 完全性：「〜について述べる」「〜を担当する」などの不完全な記述は除外
4. 禁止表現：「それ」「この」「その」「あの」「上記の」「該当の」「当該の」「その結果」「その教訓」「その事故」などの指示語は絶対使用禁止

良い例：「1955年12月19日に原子力基本法が成立した」
悪い例：「旧文部省が原子力行政を開始した」（時期不明）
悪い例：「組織体制について述べる」（内容なし）

{chunk}

1."""
    
    try:
        response = generate_response(
            prompt=prompt,
            think=think,
            max_tokens=max_tokens
        )
        return response if response else ""
    except Exception as e:
        print(f"Error in fact distillation: {e}")
        return ""

def generate_qa(fact: str, think: bool = False, max_tokens: int = None) -> str:
    """事実から質問・回答ペアを生成する"""
    prompt = f"""You are an AI assistant knowledgeable about nuclear safety regulation. Be accurate but concise in response.

Write 10 pairs of questions and answers probing the fact and statistics of the below fact about nuclear safety. :
{fact}

You can firstly generate questions in Japanese and answers that are very relevant and explicit to the fact, and then paraphrase question and answers to reach the desired 10 pairs.
回答は、ですます調でお願いします。
1. Q: サンプル質問 A: サンプル回答
2."""
    try:
        response = generate_response(
            prompt=prompt,
            think=think,
            max_tokens=max_tokens
        )
        return response if response else ""
    except Exception as e:
        print(f"Error in QA generation: {e}")
        return ""

def generate_questions_only(fact: str, think: bool = False, max_tokens: int = None) -> str:
    """事実から質問のみを生成する"""
    prompt = f"""You are an AI assistant knowledgeable about nuclear safety regulation. 

Generate 10 questions in Japanese that probe the fact and statistics of the below fact about nuclear safety:
{fact}

Only generate questions, not answers. Format as:
1. Q: 質問1
2. Q: 質問2
3. Q: 質問3
...
10. Q: 質問10"""
    try:
        response = generate_response(
            prompt=prompt,
            think=think,
            max_tokens=max_tokens
        )
        return response if response else ""
    except Exception as e:
        print(f"Error in question generation: {e}")
        return ""

async def generate_answer_with_rag(question: str, fact: str, vector_db=None, k: int = 3, think: bool = False, max_tokens: int = None) -> str:
    """質問に対してRAGを使用して回答を生成する"""
    context = ""
    if vector_db is not None:
        try:
            # 質問から関連文書を検索
            question_results = await vector_db.search_similar_texts(question, k=k)
            # ファクトからも関連文書を検索
            fact_results = await vector_db.search_similar_texts(fact, k=k)
            # ハイブリッド結果をマージ（重複除去）
            all_results = {}
            for chunk_id, text, distance in question_results:
                all_results[chunk_id] = (text, distance, "question")
            for chunk_id, text, distance in fact_results:
                if chunk_id not in all_results or distance < all_results[chunk_id][1]:
                    all_results[chunk_id] = (text, distance, "fact")
            # 距離でソートして上位k*2個を取得
            sorted_results = sorted(all_results.items(), key=lambda x: x[1][1])[:k*2]
            if sorted_results:
                context = "\n\n以下は関連する原子力規制文書の抜粋です。これを参考にして回答してください:\n"
                for i, (chunk_id, (text, distance, search_type)) in enumerate(sorted_results, 1):
                    context += f"\n[参考文書 {i}]:\n{text}\n"
                print(f"Found {len(sorted_results)} document chunks for RAG context")
        except Exception as e:
            print(f"RAG retrieval failed, proceeding without context: {e}")
    prompt = f"""You are an AI assistant knowledgeable about nuclear safety regulation. Be accurate and concise.
{context}

Based on the following fact about nuclear safety:
{fact}

Please answer the following question in Japanese using です/ます form.
IMPORTANT: If you don't have enough information to answer the question based on the provided context and fact, please respond with "[specific topic]については情報がありません。" where you replace [specific topic] with the actual topic being asked about in the question. For example:
- If asked about "反応時間", respond with "反応時間については情報がありません。"
- If asked about "統計データ", respond with "統計データについては情報がありません。"
- If asked about "測定頻度", respond with "測定頻度については情報がありません。"
Do NOT use "〇〇については情報がありません。" with the actual 〇〇 symbols.

質問: {question}

回答:"""
    try:
        response = generate_response(
            prompt=prompt,
            think=think,
            max_tokens=max_tokens
        )
        return response if response else ""
    except Exception as e:
        print(f"Error in answer generation: {e}")
        return ""

def parse_facts(llm_output: str) -> list:
    """LLM出力からfactリストを抽出する"""
    import re
    
    facts = []
    
    # 番号付きリスト形式のパターンを検索 (1., 2., 3. など)
    numbered_pattern = r'^\s*(\d+)\.?\s*(.+?)(?=^\s*\d+\.|\Z)'
    matches = re.findall(numbered_pattern, llm_output, re.MULTILINE | re.DOTALL)
    
    if matches:
        for _, fact_text in matches:
            fact = fact_text.strip()
            if fact:
                facts.append(fact)
        return facts
    
    # 改行区切りで分割して処理
    lines = llm_output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 番号や記号を除去
        # パターン: "1. ", "- ", "• ", "・ " など
        cleaned_line = re.sub(r'^[\d\-•・\s]*\.?\s*', '', line)
        
        if cleaned_line:
            facts.append(cleaned_line)
    
    return facts

def parse_questions(questions_output: str) -> list:
    """LLM出力から質問リストを抽出する"""
    import re
    questions = []
    # Q: ... 形式のパターンを検索
    question_pattern = r'(?:^\s*\d+\.?\s*)?Q:\s*(.+?)(?=(?:^\s*\d+\.?\s*)?Q:|$)'
    matches = re.findall(question_pattern, questions_output, re.MULTILINE | re.DOTALL)
    for question in matches:
        question = question.strip()
        # **、改行、余分な空白を削除
        question = re.sub(r'\*\*', '', question)  # **を削除
        question = re.sub(r'\n+', ' ', question)  # 改行を空白に変換
        question = re.sub(r'\s+', ' ', question)  # 連続空白を単一空白に変換
        question = question.strip()
        if question:
            questions.append(question)
    return questions

def create_qa_pair_json(question: str, answer: str, fact: str = "") -> str:
    """質問と回答からAlpaca形式のJSONを作成する"""
    import json
    import re
    # 回答から**、改行、余分な空白を削除
    answer = re.sub(r'\*\*', '', answer)  # **を削除
    answer = re.sub(r'\n+', ' ', answer)  # 改行を空白に変換
    answer = re.sub(r'\s+', ' ', answer)  # 連続空白を単一空白に変換
    answer = answer.strip()
    qa_entry = {
        "instruction": question,
        "input": fact,
        "output": answer
    }
    return json.dumps(qa_entry, ensure_ascii=False)

def parse_qa_pairs(qa_output: str, fact: str = "") -> list:
    """LLM出力からQ&Aペアを抽出してデータセット用のJSON文字列リストを返す"""
    import re
    import json
    qa_json_list = []
    # Q: ... A: ... 形式のパターンを検索
    qa_pattern = r'(?:^\s*\d+\.?\s*)?Q:\s*(.+?)\s*A:\s*(.+?)(?=(?:^\s*\d+\.?\s*)?Q:|$)'
    matches = re.findall(qa_pattern, qa_output, re.MULTILINE | re.DOTALL)
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        # instructionとoutputから**、改行、余分な空白を削除
        import re
        question = re.sub(r'\*\*', '', question)  # **を削除
        question = re.sub(r'\n+', ' ', question)  # 改行を空白に変換
        question = re.sub(r'\s+', ' ', question)  # 連続空白を単一空白に変換
        question = question.strip()
        answer = re.sub(r'\*\*', '', answer)  # **を削除
        answer = re.sub(r'\n+', ' ', answer)  # 改行を空白に変換
        answer = re.sub(r'\s+', ' ', answer)  # 連続空白を単一空白に変換
        answer = answer.strip()
        if question and answer:
            qa_entry = {
                "instruction": question,
                "input": fact,
                "output": answer
            }
            qa_json_list.append(json.dumps(qa_entry, ensure_ascii=False))
    return qa_json_list

def split_text(input_str, max_length):
    """テキストを句点区切りで指定長以下のチャンクに分割"""
    result = []
    current_chunk = ""
    for char in input_str:
        current_chunk += char
        if char == '。':
            if len(current_chunk) >= max_length:
                result.append(current_chunk)
                current_chunk = ""
        elif len(current_chunk) >= max_length:
            last_period_index = current_chunk.rfind('。')
            if last_period_index != -1:
                result.append(current_chunk[:last_period_index + 1])
                current_chunk = current_chunk[last_period_index + 1:]
    if current_chunk:
        result.append(current_chunk)
    return result

def process_text_file_to_alpaca(filepath: str, output_filename: str = None, max_chunks: int = None, think: bool = False, max_tokens: int = None) -> None:
    """テキストファイルを読み込み、チャンク分割してアルパカデータセットを生成"""
    if not output_filename:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
        output_filename = f"data/{timestamp}_{base_name}_facts.jsonl"
    
    print(f"Reading text file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Splitting text into chunks...")
    chunks = split_text(content, 1024)
    print(f"Found {len(chunks)} chunks")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Processing first {max_chunks} chunks")
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("")
    
    total_entries = 0
    
    for i, chunk_text in enumerate(chunks):
        try:
            chunk_text = remove_until_parenthesis(chunk_text)
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            print("Chunk preview:", chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text)
            
            # fact distillationを実行
            facts_response = fact_distillation(chunk_text, think=think, max_tokens=max_tokens)
            if not facts_response:
                print(f"Failed to get facts for chunk {i+1}")
                continue
            
            # factsをパース
            facts = parse_facts(facts_response)
            if not facts:
                print(f"No facts extracted from chunk {i+1}")
                continue
            
            # 各factからQAペアを生成してJSONLに保存
            for fact in facts:
                qa_output = generate_qa(fact, think=think, max_tokens=max_tokens)
                if qa_output:
                    qa_json_list = parse_qa_pairs(qa_output, fact)
                    for qa_json in qa_json_list:
                        with open(output_filename, "a", encoding="utf-8") as f:
                            f.write(qa_json + "\n")
            total_entries += 1
            print(f"Saved facts from chunk {i+1}")
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    print(f"Dataset generation complete. Saved {total_entries} entries to {output_filename}")

async def process_text_file_with_rag_workflow(filepath: str, output_filename: str = None, max_chunks: int = None, think: bool = False, max_tokens: int = None) -> None:
    """テキストファイルを読み込み、Q→RAG→Aのワークフローでアルパカデータセットを生成"""
    from src.core.vector_db import VectorDB
    if not output_filename:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
        output_filename = f"data/{timestamp}_{base_name}_rag_workflow.jsonl"
    print(f"Reading text file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    print("Splitting text into chunks...")
    all_chunks = split_text(content, 1024)
    print(f"Found {len(all_chunks)} chunks")
    # VectorDBに登録するチャンク数（デフォルト1000）
    db_chunks_limit = 1000 if len(all_chunks) > 1000 else len(all_chunks)
    db_chunks = all_chunks[:db_chunks_limit]
    print(f"Using first {db_chunks_limit} chunks for VectorDB")
    # 処理対象チャンク（ランダム選択またはmax_chunks制限）
    import random
    if max_chunks and max_chunks < len(all_chunks):
        # ランダムに選択
        processing_chunks = random.sample(all_chunks, max_chunks)
        print(f"Randomly selected {max_chunks} chunks for processing")
    else:
        processing_chunks = all_chunks
        print(f"Processing all {len(processing_chunks)} chunks")
    # VectorDBを初期化（古いDBとembeddingを消去して全テキストから再構築）
    print("Initializing VectorDB and clearing old data...")
    vector_db = VectorDB()
    vector_db.clear_database()
    print("Old database and embeddings cleared")
    print("Adding chunks to VectorDB...")
    await vector_db.add_text_chunks(db_chunks, source_file=filepath)
    print(f"Added {len(db_chunks)} chunks to VectorDB")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("")
    import datetime
    total_entries = 0
    qa_count = 0
    start_time = time.time()
    n_chunks = len(processing_chunks)
    def _ts():
        return datetime.datetime.now().strftime("%H:%M:%S")
    def _eta(i):
        elapsed = time.time() - start_time
        if i == 0:
            return "計算中"
        per_chunk = elapsed / i
        remaining = per_chunk * (n_chunks - i)
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        return f"{eta.strftime('%H:%M')} (残{int(remaining//60)}分)"
    for i, chunk_text in enumerate(processing_chunks):
        try:
            chunk_text = remove_until_parenthesis(chunk_text)
            pct = (i + 1) / n_chunks * 100
            print(f"[{_ts()}] chunk {i+1}/{n_chunks} ({pct:.0f}%) ETA:{_eta(i)} QA累計:{qa_count}")
            print("Chunk preview:", chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text)
            # fact distillationを実行
            facts_response = fact_distillation(chunk_text, think=think, max_tokens=max_tokens)
            if not facts_response:
                print(f"[{_ts()}] Failed to get facts for chunk {i+1}")
                continue
            # factsをパース
            facts = parse_facts(facts_response)
            if not facts:
                print(f"[{_ts()}] No facts extracted from chunk {i+1}")
                continue
            # 各factから質問を生成
            for fact in facts:
                print(f"Processing fact: {fact[:50]}...")
                # Step 1: 質問のみ生成
                questions_output = generate_questions_only(fact, think=think, max_tokens=max_tokens)
                if not questions_output:
                    print(f"[{_ts()}] Failed to generate questions for fact")
                    continue
                # Step 2: 質問をパース
                questions = parse_questions(questions_output)
                if not questions:
                    print(f"[{_ts()}] No questions extracted from fact")
                    continue
                print(f"Generated {len(questions)} questions")
                # Step 3: 各質問に対してRAGで回答生成
                for question in questions:
                    answer = await generate_answer_with_rag(question, fact, vector_db=vector_db, k=3, think=think, max_tokens=max_tokens)
                    if answer:
                        # Step 4: Q&Aペアを保存
                        qa_json = create_qa_pair_json(question, answer, fact)
                        with open(output_filename, "a", encoding="utf-8") as f:
                            f.write(qa_json + "\n")
                        qa_count += 1
                        print(f"Saved Q&A pair: Q={question[:30]}...")
            total_entries += 1
            elapsed = time.time() - start_time
            print(f"[{_ts()}] chunk {i+1}/{n_chunks} 完了 ({(i+1)/n_chunks*100:.0f}%) elapsed:{elapsed/60:.1f}分 ETA:{_eta(i+1)} QA累計:{qa_count}")
        except Exception as e:
            print(f"[{_ts()}] Error processing chunk {i+1}: {e}")
            continue
    elapsed_total = time.time() - start_time
    vector_db.close()
    print(f"[{_ts()}] RAG workflow complete. {qa_count} QA pairs, elapsed:{elapsed_total/60:.1f}分, output:{output_filename}")

def generate_dataset_distil_facts(filepath: str, output_file: str = None, max_chunks: int = None, think: bool = False, max_tokens: int = None) -> int:
    """テキストファイルからfact distillation形式のデータセットを生成する"""
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
        output_file = f"data/{timestamp}_{base_name}_facts.jsonl"
    
    print(f"Reading text file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Splitting text into chunks...")
    chunks = split_text(content, 1024)
    print(f"Found {len(chunks)} chunks")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Processing first {max_chunks} chunks")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
    
    total_entries = 0
    
    for i, chunk_text in enumerate(chunks):
        try:
            chunk_text = remove_until_parenthesis(chunk_text)
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            print("Chunk preview:", chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text)
            
            # fact distillationを実行
            facts_response = fact_distillation(chunk_text, think=think, max_tokens=max_tokens)
            if not facts_response:
                print(f"Failed to get facts for chunk {i+1}")
                continue
            
            # factsをパース
            facts = parse_facts(facts_response)
            if not facts:
                print(f"No facts extracted from chunk {i+1}")
                continue
            
            # 各factからQAペアを生成してJSONLに保存
            for fact in facts:
                qa_output = generate_qa(fact, think=think, max_tokens=max_tokens)
                if qa_output:
                    qa_json_list = parse_qa_pairs(qa_output, fact)
                    for qa_json in qa_json_list:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(qa_json + "\n")
            total_entries += 1
            print(f"Saved facts from chunk {i+1}")
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    print(f"Dataset generation complete. Saved {total_entries} entries to {output_file}")
    return total_entries

def generate_dataset_file_chunks(filepath: str, output_file: str = None, max_chunks: int = None, think: bool = False, max_tokens: int = None) -> int:
    """指定されたファイルをチャンクに分割してAlpaca形式のデータセットを生成する"""
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
        output_file = f"data/{timestamp}_{base_name}_active_reading.jsonl"
    
    print(f"Reading text file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Splitting text into chunks...")
    chunks = split_text(content, 1024)
    print(f"Found {len(chunks)} chunks")
    
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Processing first {max_chunks} chunks")
    
    total_entries = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
    
    for chunk_idx, chunk_text in enumerate(chunks):
        try:
            chunk = remove_until_parenthesis(chunk_text)
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
            print("Chunk:", chunk[:100] + "..." if len(chunk) > 100 else chunk)
            
            strategies_response = generate_task_specific_strategies(chunk, think=think, max_tokens=max_tokens)
            if not strategies_response:
                print(f"Failed to generate strategies for chunk {chunk_idx + 1}")
                continue
            
            strategies = parse_strategies(strategies_response)
            print(f"Parsed {len(strategies)} strategies for chunk {chunk_idx + 1}")
            
            for i, strategy in enumerate(strategies):
                try:
                    print(f"Applying strategy {i+1} to chunk {chunk_idx + 1}...")
                    result = apply_strategy_to_chunk(strategy, chunk, think=think, max_tokens=max_tokens)
                    if not result:
                        print(f"Failed to apply strategy {i+1} to chunk {chunk_idx + 1}")
                        continue
                    
                    instruction_with_document = f"""Here's a learning strategy:
{strategy}
Apply this strategy to the following document:
<document>
{chunk}
</document>"""
                    alpaca_entry = {
                        "instruction": instruction_with_document,
                        "input": "",
                        "output": result
                    }
                    append_to_jsonl(alpaca_entry, output_file)
                    total_entries += 1
                    print(f"Saved entry {total_entries}")
                except Exception as e:
                    print(f"Error applying strategy {i+1} to chunk {chunk_idx + 1}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            continue
    
    print(f"Generated {total_entries} alpaca dataset entries saved to {output_file}")
    return total_entries

if __name__ == '__main__':
    import argparse
    import sys
    
    if len(sys.argv) == 1:
        print("Usage: python augument.py <filepath> [options]")
        print()
        print("Generate dataset from text file")
        print()
        print("Required arguments:")
        print("  filepath              Path to input text file")
        print()
        print("Optional arguments:")
        print("  -h, --help            Show this help message and exit")
        print("  -o, --output OUTPUT   Output JSONL file path (default: auto-generated)")
        print("  -c, --max-chunks CHUNKS  Maximum number of chunks to process (default: all)")
        print("  -t, --max-tokens TOKENS  Maximum tokens per response (default: 2048)")
        print("  --mode MODE          Dataset type: 'facts' for fact distillation or 'active' for active reading (default: active)")
        print()
        print("Examples:")
        print("  python augument.py data/250304_kangaekata.txt")
        print("  python augument.py data/text.txt --max-chunks 10 --mode facts")
        print("  python augument.py data/text.txt --output data/custom.jsonl --mode active")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Generate dataset from text file')
    parser.add_argument('filepath', help='Path to input text file')
    parser.add_argument('--output', '-o', help='Output JSONL file path (optional)')
    parser.add_argument('--max-chunks', '-c', type=int, help='Maximum number of chunks to process')
    parser.add_argument('--max-tokens', '-t', type=int, default=2048, help='Maximum tokens per response')
    parser.add_argument('--mode', choices=['facts', 'active'], default='active', 
                       help='Dataset type: facts for fact distillation, active for active reading (default: active)')
    
    args = parser.parse_args()
    
    # モードに応じてデータセット生成関数を選択
    if args.mode == 'facts':
        print("🔬 Generating fact distillation dataset...")
        asyncio.run(process_text_file_with_rag_workflow(
            filepath=args.filepath,
            output_filename=args.output,
            max_chunks=args.max_chunks,
            think=False,
            max_tokens=args.max_tokens
        ))
    elif args.mode == 'active':
        print("📚 Generating active reading dataset...")
        generate_dataset_file_chunks(
            filepath=args.filepath,
            output_file=args.output,
            max_chunks=args.max_chunks,
            think=False,
            max_tokens=args.max_tokens
        )
