import os
import sqlite3
import faiss
import numpy as np
import httpx
from typing import List, Tuple, Optional

class VectorDB:
    """FAISSとSQLite3を使用したベクトルデータベース"""

    def __init__(self, db_path: str = "data/text_vectors.db", index_path: str = "data/text_vectors.faiss", embedding_dim: int = 384):
        """
        初期化

        Args:
            db_path: SQLite3データベースのパス
            index_path: FAISSインデックスのパス
            embedding_dim: 埋め込みベクトルの次元数
        """
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._init_database()
        self._init_faiss_index()

    def _init_database(self):
        """SQLite3データベースの初期化"""
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_file TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def _init_faiss_index(self):
        """FAISSインデックスの初期化"""
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
        else:
            print(f"Creating new FAISS index at {self.index_path}")
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
            faiss.write_index(index, self.index_path)

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        テキストから埋め込みベクトルを取得

        Args:
            text: 埋め込みを取得するテキスト

        Returns:
            埋め込みベクトル (numpy配列) または None
        """
        from src.config.settings import Config
        url = Config.E5_EMBEDDING_URL
        payload = {'query': text}
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    embedding = response.json()
                    return np.array(embedding, dtype=np.float32)
                else:
                    print(f"Error: Embedding request failed with status code {response.status_code}")
                    return None
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    async def add_text_chunks(self, chunks: List[str], source_file: str = None):
        """
        テキストチャンクをデータベースとFAISSインデックスに追加

        Args:
            chunks: テキストチャンクのリスト
            source_file: ソースファイル名
        """
        index = faiss.read_index(self.index_path)
        for i, chunk in enumerate(chunks):
            self.cur.execute(
                'INSERT INTO text_chunks (text, source_file, chunk_index) VALUES (?, ?, ?)',
                (chunk, source_file, i)
            )
            chunk_id = self.cur.lastrowid
            embedding = await self.get_embedding(chunk)
            if embedding is not None:
                index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([chunk_id], dtype=np.int64))
                print(f"Added chunk {i+1}/{len(chunks)} (ID: {chunk_id})")
            else:
                print(f"Failed to add embedding for chunk {i+1}")
        faiss.write_index(index, self.index_path)
        self.conn.commit()
        print(f"Total vectors in index: {index.ntotal}")

    async def search_similar_texts(self, query: str, k: int = 5) -> List[Tuple[int, str, float]]:
        """
        クエリに類似したテキストを検索

        Args:
            query: 検索クエリ
            k: 取得する結果数

        Returns:
            (chunk_id, text, distance) のタプルのリスト
        """
        try:
            index = faiss.read_index(self.index_path)
            query_embedding = await self.get_embedding(query)
            if query_embedding is None:
                return []
            D, I = index.search(np.array([query_embedding], dtype=np.float32), k)
            results = []
            for i in range(k):
                chunk_id = int(I[0][i])
                distance = float(D[0][i])
                result = self.cur.execute(
                    'SELECT text FROM text_chunks WHERE chunk_id = ?',
                    (chunk_id,)
                ).fetchone()
                if result:
                    results.append((chunk_id, result[0], distance))
            return results
        except Exception as e:
            print(f"Error searching similar texts: {e}")
            return []

    def clear_database(self):
        """データベースとインデックスをクリア"""
        self.cur.execute('DELETE FROM text_chunks')
        self.conn.commit()
        index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        faiss.write_index(index, self.index_path)
        print("Database and index cleared")

    def get_stats(self) -> dict:
        """データベースの統計情報を取得"""
        total_chunks = self.cur.execute('SELECT COUNT(*) FROM text_chunks').fetchone()[0]
        index = faiss.read_index(self.index_path)
        total_vectors = index.ntotal
        return {
            'total_chunks': total_chunks,
            'total_vectors': total_vectors
        }

    def close(self):
        """データベース接続を閉じる"""
        self.conn.close()
