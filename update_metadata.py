import os
import json

# 读取原始metadata.json
metadata_path = os.path.join('vector_db_manual', 'metadata.json')

try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    print(f"读取了 {len(metadata_list)} 条元数据记录")
    
    # 更新路径信息
    for item in metadata_list:
        if "path" in item:
            original_path = item["path"]
            # 提取文件名
            filename = os.path.basename(original_path)
            # 构建新路径
            new_path = os.path.join("manual_chunks", filename)
            item["path"] = new_path
    
    # 保存更新后的metadata.json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    print(f"已更新 metadata.json 中的路径信息")
    
except Exception as e:
    print(f"处理metadata.json时出错: {e}")