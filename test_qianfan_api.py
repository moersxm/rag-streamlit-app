import os
import json
import time
import datetime
import uuid

def test_with_appbuilder_sdk():
    """
    使用AppBuilder SDK测试文档上传和解析流程
    """
    print("\n" + "="*60)
    print("使用AppBuilder SDK进行文档上传和解析测试")
    print("="*60)
    
    try:
        # 尝试导入AppBuilder SDK
        import appbuilder
    except ImportError:
        print("AppBuilder SDK未安装。请先运行: pip install --upgrade appbuilder-sdk")
        return
    
    # 设置环境中的TOKEN
    os.environ["APPBUILDER_TOKEN"] = "bce-v3/ALTAK-tTXXQUFQTzD0wmpZaZcw8/6339a986fa067a766bb5cb45e94ec619443829d3"
    
    # 应用ID
    app_id = "520517e8-c557-4715-847f-293430dedf59"
    
    print(f"\nAppBuilder SDK配置:")
    print(f"应用ID: {app_id}")
    print(f"APPBUILDER_TOKEN: {'已设置' if 'APPBUILDER_TOKEN' in os.environ else '未设置'}")
    
    # 初始化AppBuilder客户端
    print("\n正在初始化AppBuilder客户端...")
    app_builder_client = appbuilder.AppBuilderClient(app_id)
    print("初始化成功!")
    
    # 获取测试文件路径
    file_path = input("\n请输入要测试的文件路径（例如: test.pdf）: ")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在!")
        return
    
    # 步骤1: 创建会话
    print("\n1. 创建会话")
    print("-"*40)
    
    try:
        conversation_id = app_builder_client.create_conversation()
        print(f"会话创建成功! Conversation ID: {conversation_id}")
    except Exception as e:
        print(f"创建会话失败: {str(e)}")
        return
    
    # 步骤2: 上传文件
    print("\n2. 上传文件")
    print("-"*40)
    
    file_name = os.path.basename(file_path)
    print(f"上传文件: {file_name}")
    
    try:
        file_id = app_builder_client.upload_file(conversation_id, file_path)
        print(f"文件上传成功! File ID: {file_id}")
    except Exception as e:
        print(f"文件上传失败: {str(e)}")
        return
    
    # 步骤3: 提取文件内容
    print("\n3. 提取文件内容")
    print("-"*40)
    
    try:
        result = app_builder_client.extract_file(conversation_id, file_id)
        
        if "content" in result:
            content = result["content"]
            print(f"内容提取成功! 长度: {len(content)}字符")
            print(f"内容预览 (前300个字符): {content[:300]}...")
            
            # 保存提取的文本到文件
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"extracted_{timestamp}_{file_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"提取的文本已保存到: {output_file}")
        else:
            print("内容提取失败: 响应中没有content字段")
            print(f"完整响应: {result}")
            return
    except Exception as e:
        print(f"提取文件内容失败: {str(e)}")
        return
    
    # 步骤4: 文档解析
    print("\n4. 文档解析")
    print("-"*40)
    
    # 构建解析提示词
    prompt = f"""请作为政府采购和PPP项目专家，对以下文档内容进行专业解析，提取关键信息并按以下结构输出：
    
    # 文档解析报告
    
    ## 1. 基本信息
    - 文档名称: {file_name}
    - 文档类型: [政策文件/法规/指南/其他]
    - 发布机构: [提取文档中的发布机构]
    - 文件编号: [提取文档中的文件编号，如有]
    
    ## 2. 主要内容摘要
    [200字以内的文档主要内容概述]
    
    ## 3. 关键条款和政策要点
    [列出3-5个核心条款或政策要点，使用短句概括]
    
    ## 4. 适用范围
    [描述该文档的适用对象和适用情境]
    
    ## 5. 实施要求
    [描述文档中提到的实施要求、期限或程序性规定]
    
    ## 6. 与其他政策的关系
    [分析该文档与其他相关政策的关系，如有提及]
    
    ## 7. 专家评析
    [从专业角度分析该文档的重要性和实际应用价值]
    
    文档内容:
    {content[:3000]}  # 截取前3000字符，避免超出模型token限制
    """
    
    try:
        print(f"开始文档解析...")
        start_time = time.time()
        
        # 发送解析请求
        resp = app_builder_client.run(conversation_id, prompt)
        parse_result = resp.content.answer
        
        end_time = time.time()
        print(f"解析完成，耗时: {end_time - start_time:.2f}秒")
        
        # 保存解析结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
        os.makedirs(output_dir, exist_ok=True)
        parse_file = os.path.join(output_dir, f"analysis_{timestamp}_{file_name}.md")
        with open(parse_file, 'w', encoding='utf-8') as f:
            f.write(f"# {file_name} 解析报告\n\n")
            f.write(f"**解析时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(parse_result)
            
        print(f"解析结果已保存到: {parse_file}")
        
        # 显示解析结果预览
        print("\n解析结果预览:")
        print("-"*40)
        print(parse_result[:500] + "..." if len(parse_result) > 500 else parse_result)
        
    except Exception as e:
        print(f"文档解析失败: {str(e)}")
        return
    
    print("\n" + "="*60)
    print("测试流程完成!")
    print("="*60)

def test_with_direct_api():
    """
    直接使用API测试文档上传和解析流程
    """
    print("\n" + "="*60)
    print("使用直接API调用进行文档上传和解析测试")
    print("="*60)
    
    import requests
    
    # API配置
    base_url = "https://qianfan.baidubce.com"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-tTXXQUFQTzD0wmpZaZcw8/6339a986fa067a766bb5cb45e94ec619443829d3'
    }
    app_id = "520517e8-c557-4715-847f-293430dedf59"  # 应用ID
    
    # 步骤1: 创建会话
    print("\n1. 创建会话")
    print("-"*40)
    
    conversation_url = f"{base_url}/v2/app/conversation"
    payload = json.dumps({
        "app_id": app_id
    })
    
    print(f"请求URL: {conversation_url}")
    print(f"请求体: {payload}")
    
    try:
        response = requests.post(
            conversation_url,
            headers=headers,
            data=payload
        )
        
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n响应内容:")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            
            # 提取会话ID
            if "conversation_id" in result:
                conversation_id = result["conversation_id"]
                print(f"\n会话创建成功! Conversation ID: {conversation_id}")
            else:
                print("\n创建会话失败: 响应中无会话ID")
                return
        else:
            print("\n响应内容:")
            print(response.text)
            print(f"\n创建会话请求失败: HTTP状态码 {response.status_code}")
            return
            
    except Exception as e:
        print(f"\n创建会话时发生错误: {e}")
        return
    
    # 步骤2: 上传文件
    print("\n2. 上传文件")
    print("-"*40)
    
    # 获取测试文件
    file_path = input("请输入要测试的文件路径（例如: test.pdf）: ")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在!")
        return
    
    # 读取文件
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 准备上传文件
    file_name = os.path.basename(file_path)
    upload_url = f"{base_url}/v2/app/conversation/file/upload"
    
    # 文件上传需要使用multipart/form-data，因此不使用Content-Type头
    upload_headers = {
        'Authorization': headers['Authorization']
    }
    
    print(f"\n开始上传文件 '{file_name}'...")
    print(f"请求URL: {upload_url}")
    print(f"会话ID: {conversation_id}")
    
    try:
        # 发送请求
        response = requests.post(
            upload_url,
            headers=upload_headers,
            files={'file': (file_name, file_data)},
            data={
                'app_id': app_id,
                'conversation_id': conversation_id
            }
        )
        
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n响应内容:")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            
            if 'id' in result:
                file_id = result['id']
                print(f"\n文件上传成功! File ID: {file_id}")
            else:
                print(f"\n文件上传失败: {result.get('error_msg', '未知错误')}")
                return
        else:
            print("\n响应内容:")
            print(response.text)
            print(f"\n文件上传请求失败: HTTP状态码 {response.status_code}")
            return
    
    except Exception as e:
        print(f"\n上传文件时发生错误: {e}")
        return
    
    # 步骤3: 提取文件内容
    print("\n3. 提取文件内容")
    print("-"*40)
    
    extract_url = f"{base_url}/v2/app/conversation/file/extract"
    
    # 恢复使用JSON内容类型
    extract_headers = {
        'Content-Type': 'application/json',
        'Authorization': headers['Authorization']
    }
    
    extract_payload = json.dumps({
        "id": file_id,
        "app_id": app_id,
        "conversation_id": conversation_id
    })
    
    print(f"请求URL: {extract_url}")
    print(f"请求体: {extract_payload}")
    
    try:
        response = requests.post(
            extract_url,
            headers=extract_headers,
            data=extract_payload
        )
        
        print(f"\n响应状态码: {response.status_code}")
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if response.status_code == 200:
            result = response.json()
            print("\n响应内容片段:")
            
            if "content" in result:
                content = result["content"]
                # 只打印内容的前300个字符，避免过长
                print(f"提取的内容 (前300个字符): {content[:300]}...")
                print("\n文件内容提取成功!")
                
                # 保存提取的内容到文件
                output_file = os.path.join(output_dir, f"extracted_{timestamp}_{file_name}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"内容已保存到文件: {output_file}")
                
                # 开始文档解析
                print("\n4. 文档解析")
                print("-"*40)
                
                # 构建解析提示词
                prompt = f"""请作为政府采购和PPP项目专家，对以下文档内容进行专业解析，提取关键信息并按以下结构输出：
                
                # 文档解析报告
                
                ## 1. 基本信息
                - 文档名称: {file_name}
                - 文档类型: [政策文件/法规/指南/其他]
                - 发布机构: [提取文档中的发布机构]
                - 文件编号: [提取文档中的文件编号，如有]
                
                ## 2. 主要内容摘要
                [200字以内的文档主要内容概述]
                
                ## 3. 关键条款和政策要点
                [列出3-5个核心条款或政策要点，使用短句概括]
                
                ## 4. 适用范围
                [描述该文档的适用对象和适用情境]
                
                ## 5. 实施要求
                [描述文档中提到的实施要求、期限或程序性规定]
                
                ## 6. 与其他政策的关系
                [分析该文档与其他相关政策的关系，如有提及]
                
                ## 7. 专家评析
                [从专业角度分析该文档的重要性和实际应用价值]
                
                文档内容:
                {content[:3000]}  # 截取前3000字符，避免超出模型token限制
                """
                
                chat_url = f"{base_url}/v2/app/conversation/message"
                chat_payload = json.dumps({
                    "conversation_id": conversation_id,
                    "app_id": app_id,
                    "query": prompt
                })
                
                print(f"开始文档解析...")
                start_time = time.time()
                
                chat_response = requests.post(
                    chat_url,
                    headers=headers,
                    data=chat_payload
                )
                
                if chat_response.status_code == 200:
                    chat_result = chat_response.json()
                    
                    if "result" in chat_result:
                        parse_result = chat_result["result"]
                        
                        end_time = time.time()
                        print(f"解析完成，耗时: {end_time - start_time:.2f}秒")
                        
                        # 保存解析结果
                        parse_file = os.path.join(output_dir, f"analysis_{timestamp}_{file_name}.md")
                        with open(parse_file, 'w', encoding='utf-8') as f:
                            f.write(f"# {file_name} 解析报告\n\n")
                            f.write(f"**解析时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            f.write(parse_result)
                            
                        print(f"解析结果已保存到: {parse_file}")
                        
                        # 显示解析结果预览
                        print("\n解析结果预览:")
                        print("-"*40)
                        print(parse_result[:500] + "..." if len(parse_result) > 500 else parse_result)
                    else:
                        print(f"文档解析失败: 响应中没有结果")
                else:
                    print(f"文档解析请求失败: HTTP状态码 {chat_response.status_code}")
                    print(chat_response.text)
                
            else:
                print(json.dumps(result, indent=4, ensure_ascii=False))
                print(f"\n内容提取失败: {result.get('error_msg', '未知错误')}")
        else:
            print("\n响应内容:")
            print(response.text)
            print(f"\n提取内容请求失败: HTTP状态码 {response.status_code}")
    
    except Exception as e:
        print(f"\n提取文件内容时发生错误: {e}")

if __name__ == "__main__":
    print("\n选择测试方式:")
    print("1. 使用AppBuilder SDK测试")
    print("2. 使用直接API调用测试")
    
    choice = input("请输入选择 (1/2): ")
    
    if choice == "1":
        test_with_appbuilder_sdk()
    elif choice == "2":
        test_with_direct_api()
    else:
        print("无效选择，退出程序")
    
    print("\n" + "="*50)
    input("按Enter键退出...")