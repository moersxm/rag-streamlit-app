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
            # 从路径中提取文件名部分
            parts = item["path"].split('\\')
            
            # 找到manual_chunks所在的位置
            try:
                manual_chunks_index = parts.index('manual_chunks')
                # 构建新路径: manual_chunks之后的所有部分
                relative_parts = parts[manual_chunks_index:]
                new_path = os.path.join(*relative_parts)
                item["path"] = new_path
            except ValueError:
                # 如果找不到manual_chunks，使用文件名作为备选方案
                filename = parts[-1] if parts else item["filename"]
                folder = item.get("section_title", "").split()[0] if item.get("section_title") else ""
                if folder:
                    new_path = os.path.join("manual_chunks", folder, filename)
                else:
                    new_path = os.path.join("manual_chunks", filename)
                item["path"] = new_path
    
    # 保存更新后的metadata.json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    print(f"已更新 metadata.json 中的路径信息为相对路径")
    
except Exception as e:
    print(f"处理metadata.json时出错: {e}")