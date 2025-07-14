import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time

class RAGSystem:
    def __init__(self, vector_db_path="vector_db_manual"):
        self.vector_db_path = vector_db_path
        
        self.api_url = "https://qianfan.baidubce.com/v2/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer bce-v3/ALTAK-tTXXQUFQTzD0wmpZaZcw8/6339a986fa067a766bb5cb45e94ec619443829d3',
            'appid': 'app-0uIqZTDX'
        }
        self.model = "ernie-3.5-8k"
        
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"嵌入向量维度: {self.embedding_dim}")
        
        self._load_vector_db()
    
    def _load_vector_db(self):
        """加载或创建向量数据库"""
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        try:
            # 检查元数据文件是否存在
            if not os.path.exists(metadata_path):
                print(f"元数据文件不存在: {metadata_path}，将创建空的元数据文件")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                self.metadata_list = []
                self.texts = []
            else:
                # 加载元数据
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata_list = json.load(f)
                print(f"已加载 {len(self.metadata_list)} 条元数据记录")
                
                # 加载文本内容
                self.texts = []
                for item in self.metadata_list:
                    text_path = item.get("path")
                    if text_path and os.path.exists(text_path):
                        try:
                            with open(text_path, "r", encoding="utf-8") as f:
                                self.texts.append(f.read())
                        except Exception as e:
                            print(f"读取文件失败 {text_path}: {e}")
                            self.texts.append("")
                    else:
                        print(f"文件不存在: {text_path}")
                        self.texts.append("")
            
            # 检查索引文件是否存在
            index_path = os.path.join(self.vector_db_path, "index.faiss")
            if os.path.exists(index_path):
                try:
                    print(f"从文件加载向量索引: {index_path}")
                    self.index = faiss.read_index(index_path)
                    print(f"已加载向量索引，维度: {self.index.d}")
                except Exception as e:
                    print(f"加载索引失败: {e}")
                    print("将创建新的索引")
                    self._create_vector_index()
            else:
                print(f"索引文件不存在: {index_path}")
                print("将创建新的索引")
                self._create_vector_index()
        
        except Exception as e:
            raise RuntimeError(f"加载向量数据库失败: {str(e)}")

    def _create_vector_index(self):
        """创建新的向量索引"""
        print("创建新的向量索引")
        if not self.texts or len(self.texts) == 0:
            print("警告: 没有文本数据，将创建空索引")
            # 创建一个空的索引
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            return
        
        # 从文本创建向量
        try:
            print(f"为 {len(self.texts)} 条文本创建向量...")
            vectors = []
            for text in self.texts:
                if text.strip():  # 确保文本不为空
                    vector = self.embedding_model.encode(text)
                    vectors.append(vector)
                else:
                    # 对于空文本，使用零向量
                    vectors.append(np.zeros(self.embedding_dim))
            
            # 转换为numpy数组
            vectors = np.array(vectors).astype('float32')
            
            # 创建并保存索引
            print(f"创建FAISS索引，向量数量: {len(vectors)}, 维度: {self.embedding_dim}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if len(vectors) > 0:
                self.index.add(vectors)
            
            # 保存索引到文件
            index_path = os.path.join(self.vector_db_path, "index.faiss")
            print(f"保存索引到: {index_path}")
            faiss.write_index(self.index, index_path)
            print("索引创建和保存成功")
        
        except Exception as e:
            print(f"创建索引失败: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.encode(query).reshape(1, -1).astype('float32')
        
        if query_vector.shape[1] != self.embedding_dim:
            raise ValueError(f"查询向量维度 ({query_vector.shape[1]}) 与索引维度 ({self.embedding_dim}) 不匹配")
            
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]
                
                text = self.texts[idx] if idx < len(self.texts) else ""
                
                max_text_length = 800
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                
                distance = float(distances[0][i])
                similarity = 1.0 / (1.0 + distance)
                
                result = {
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": similarity
                }
                results.append(result)
        
        return results
    
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        prompt = self._build_prompt(query, contexts)
        
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "web_search": {
                "enable": False,
                "enable_citation": False,
                "enable_trace": False
            }
        }, ensure_ascii=False)
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                data=payload.encode("utf-8")
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"API响应键: {list(result.keys())}")
            
            if "result" in result:
                return result["result"]
            elif "response" in result:
                return result["response"]
            elif "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API响应结构: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return "无法解析API返回结果，请查看日志了解详情。"
        except Exception as e:
            print(f"API调用失败: {e}")
            import traceback
            traceback.print_exc()
            return f"生成回答时出错: {e}"
    
    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        prompt = "你是一个专业的政府采购和PPP项目顾问，请基于以下提供的参考文档，回答用户的问题。\n\n"
        
        for i, context in enumerate(contexts):
            metadata = context["metadata"]
            title = metadata.get("section_title") or metadata.get("title") or "未知文档"
            source = metadata.get("path", "未知来源").split("\\")[-1]
            
            prompt += f"参考文档[{i+1}]: {title}\n"
            prompt += f"来源: {source}\n"
            prompt += f"内容:\n{context['text']}\n\n"
        
        prompt += f"用户问题: {query}\n\n"
        prompt += "请基于提供的参考文档内容回答用户问题。如果参考文档中没有足够信息回答问题，请明确说明。回答应当专业、准确、简洁，并尽可能引用政策法规依据。"
        
        return prompt
    
    def answer(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        retrieval_start = time.time()
        contexts = self.retrieve(query)
        retrieval_time = time.time() - retrieval_start
        
        has_good_match = any(ctx.get("similarity", 0) > 0.3 for ctx in contexts)
        
        if not contexts:
            return {
                "answer": "抱歉，知识库中没有找到与您问题相关的信息。",
                "sources": [],
                "metrics": {
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "total_time": time.time() - start_time
                }
            }
        
        if not has_good_match:
            print("警告: 检索到的文档相关性较低")
        
        generation_start = time.time()
        answer = self.generate(query, contexts)
        generation_time = time.time() - generation_start
        
        sources = []
        for context in contexts:
            metadata = context["metadata"]
            source = {
                "title": metadata.get("section_title") or metadata.get("title") or "未知标题",
                "path": metadata.get("path", "未知来源"),
                "similarity": context["similarity"]
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": time.time() - start_time
            }
        }