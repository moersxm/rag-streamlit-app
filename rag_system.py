import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time
import streamlit as st  # 添加streamlit导入


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
        
        # 修改为调用静态方法以利用缓存
        self.embedding_model = self._load_embedding_model_cached()
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"嵌入向量维度: {self.embedding_dim}")
        
        self._load_vector_db()
    
    # 将方法拆分为两部分：静态缓存方法和实例方法
    @staticmethod
    @st.cache_resource
    def _load_embedding_model_cached():
        """缓存加载embedding模型，避免重复加载并处理meta tensor问题"""
        # 确保缓存目录存在
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 注意：这里先导入SentenceTransformer，确保它在后面可用
            from sentence_transformers import SentenceTransformer
            
            # 导入torch
            import torch
            
            # 设置环境变量，避免一些潜在问题
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # 先检查本地是否有缓存的模型
            local_model_path = os.path.join(cache_dir, "paraphrase-multilingual-MiniLM-L12-v2")
            
            print(f"尝试加载嵌入模型...")
            
            # 明确指定设备，避免meta tensor问题
            device = 'cpu'  # 在云环境中，通常只有CPU可用
            print(f"使用设备: {device}")
            
            # 确保模型直接加载到指定设备，避免后续的设备转移
            model = SentenceTransformer(
                model_name_or_path='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                cache_folder=cache_dir,
                device=device
            )
            
            # 确保模型已经初始化
            _ = model.encode("测试句子", convert_to_numpy=True)
            
            print("嵌入模型加载成功!")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果标准加载失败，尝试一种更安全的加载方式
            try:
                print("尝试备用加载方法...")
                
                # 使用不同的模型或更安全的加载方式
                from sentence_transformers import SentenceTransformer  # 再次导入，以防前面的导入失败
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制使用CPU
                
                # 使用较小的模型，可能会提高成功率
                model = SentenceTransformer('distiluse-base-multilingual-cased-v1', 
                                            device='cpu',
                                            cache_folder=cache_dir)
                
                print("备用模型加载成功!")
                return model
            except Exception as e2:
                print(f"备用加载方法也失败: {e2}")
                traceback.print_exc()
                
                # 如果模型加载失败，提供一个非常简单的替代方案
                print("使用非常简单的向量化模型作为最后手段...")
                
                # 创建一个简单的向量化函数作为备用
                class SimpleEmbedder:
                    def __init__(self):
                        self.dimension = 384  # 使用与原始模型相同的维度
                        
                    def encode(self, texts, convert_to_numpy=True):
                        if isinstance(texts, str):
                            texts = [texts]
                        
                        # 创建随机向量（仅用于测试，实际应用中效果不佳）
                        import numpy as np
                        np.random.seed(42)  # 固定随机种子以获得一致的结果
                        
                        vectors = np.random.rand(len(texts), self.dimension).astype('float32')
                        # 标准化向量
                        from sklearn.preprocessing import normalize
                        vectors = normalize(vectors, norm='l2')
                        
                        return vectors[0] if len(texts) == 1 and convert_to_numpy else vectors
                    
                    def get_sentence_embedding_dimension(self):
                        return self.dimension
                
                return SimpleEmbedder()
    
    def _load_vector_db(self):
        """加载向量数据库，处理不同平台的路径差异和文件不存在的情况"""
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
            print(f"已加载 {len(self.metadata_list)} 条元数据记录")
        except FileNotFoundError:
            print(f"元数据文件不存在: {metadata_path}")
            # 创建一个空的元数据列表
            self.metadata_list = []
        except Exception as e:
            print(f"加载元数据失败: {e}")
            self.metadata_list = []
        
        # 加载文本内容
        self.texts = []
        file_errors = []
        
        for item in self.metadata_list:
            text_path = item.get("path")
            if not text_path:
                self.texts.append("")
                continue
            
            # 修复路径，将Windows路径分隔符替换为当前系统的路径分隔符
            text_path = text_path.replace('\\', os.sep)
            
            # 尝试几种可能的路径
            possible_paths = [
                text_path,  # 原始路径
                os.path.join(os.path.dirname(os.path.dirname(self.vector_db_path)), text_path),  # 相对于项目根目录
                os.path.join(self.vector_db_path, os.path.basename(text_path)),  # 在向量数据库目录中寻找文件
                os.path.join("manual_chunks", os.path.basename(text_path))  # 在manual_chunks目录中寻找文件
            ]
            
            file_found = False
            for p_path in possible_paths:
                try:
                    if os.path.exists(p_path):
                        with open(p_path, "r", encoding="utf-8") as f:
                            self.texts.append(f.read())
                        file_found = True
                        # 更新元数据中的路径为正确路径
                        item["path"] = p_path
                        break
                except Exception as e:
                    continue
            
            if not file_found:
                self.texts.append("")  # 添加空文本
                file_errors.append(f"无法读取文件: {text_path}")
        
        # 报告文件错误
        if file_errors:
            print(f"警告: 无法读取 {len(file_errors)} 个文件")
            for err in file_errors[:5]:  # 只显示前5个错误
                print(f"  - {err}")
            if len(file_errors) > 5:
                print(f"  - 以及其他 {len(file_errors) - 5} 个文件")
        
        # 加载向量索引
        index_path = os.path.join(self.vector_db_path, "index.faiss")
        if os.path.exists(index_path):
            try:
                print("正在加载预计算的向量索引...")
                self.index = faiss.read_index(index_path)
                print(f"已加载向量索引，维度: {self.index.d}")
            except Exception as e:
                print(f"加载索引失败: {e}")
                print("尝试创建新索引...")
                self._create_vector_index()
        else:
            print("索引文件不存在，尝试创建新索引...")
            self._create_vector_index()
        
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
        """检索与查询相关的文档"""
        if not hasattr(self, 'index') or self.index is None:
            print("警告: 索引未初始化，无法执行检索")
            return []
        
        try:
            start_time = time.time()
            
            # 向量化查询
            if self.embedding_model:
                query_vector = self.embedding_model.encode(query).reshape(1, -1).astype('float32')
            else:
                # 如果没有embedding模型，使用随机向量（仅用于测试）
                print("警告: 没有embedding模型，使用随机向量")
                import numpy as np
                query_vector = np.random.rand(1, self.index.d).astype('float32')
                # 标准化向量
                faiss.normalize_L2(query_vector)
            
            # 检查向量维度是否与索引匹配
            if query_vector.shape[1] != self.index.d:
                print(f"警告: 查询向量维度 ({query_vector.shape[1]}) 与索引维度 ({self.index.d}) 不匹配")
                # 尝试调整向量维度
                if query_vector.shape[1] > self.index.d:
                    query_vector = query_vector[:, :self.index.d]
                else:
                    # 扩展向量维度
                    padded = np.zeros((1, self.index.d), dtype='float32')
                    padded[:, :query_vector.shape[1]] = query_vector
                    query_vector = padded
                # 重新标准化
                faiss.normalize_L2(query_vector)
            
            # 搜索相似文档
            distances, indices = self.index.search(query_vector, min(top_k, len(self.texts)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.texts) and self.texts[idx].strip():
                    metadata = self.metadata_list[idx] if idx < len(self.metadata_list) else {}
                    
                    # 修改返回结果的结构，使其与_build_prompt方法兼容
                    result = {
                        "content": self.texts[idx],
                        "text": self.texts[idx],  # 添加text键作为备份
                        "score": float(1 - distances[0][i]),  # 转换为相似度分数
                        "similarity": float(1 - distances[0][i]),  # 添加similarity键作为备份
                        "title": metadata.get("section_title", os.path.basename(metadata.get("path", ""))),
                        "path": metadata.get("path", "")
                    }
                    results.append(result)
            
            retrieval_time = time.time() - start_time
            print(f"检索耗时: {retrieval_time:.3f}秒")
            
            return results
        except Exception as e:
            print(f"检索错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
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
                "enable": True,
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
            # 检查context结构，确保我们访问的键确实存在
            title = "未知文档"
            source = "未知来源"
            
            # 直接从context中获取相关信息，不依赖于"metadata"键
            if "title" in context:
                title = context["title"]
            elif "content" in context and isinstance(context["content"], str) and len(context["content"]) > 50:
                # 使用内容的前50个字符作为标题
                title = context["content"][:50] + "..."
                
            if "path" in context:
                source_path = context["path"]
                # 提取文件名作为来源
                source = os.path.basename(source_path) if source_path else "未知来源"
            
            content = context.get("content", "")
            if not content and "text" in context:
                content = context["text"]
                
            prompt += f"参考文档[{i+1}]: {title}\n"
            prompt += f"来源: {source}\n"
            prompt += f"内容:\n{content}\n\n"
        
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
        
        # 修复这里的metadata访问问题
        sources = []
        for context in contexts:
            # 直接从context中获取信息，不再依赖metadata键
            source = {
                "title": context.get("title", "未知标题"),
                "path": context.get("path", "未知来源"),
                "similarity": context.get("similarity", context.get("score", 0))
            }
            sources.append(source)
        
        # 如果相关性较低，添加免责声明
        if not has_good_match:
            answer = f"【注意：知识库中没有找到与您问题直接相关的信息，以下回答基于AI的通用知识，可能不完全准确】\n\n{answer}"
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": time.time() - start_time
            }
        }