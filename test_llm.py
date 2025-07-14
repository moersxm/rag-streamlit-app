import requests
import json


def main():
        
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    
    payload = json.dumps({
        "model": "ernie-3.5-8k",
        "messages": [
            {
                "role": "user",
                "content": "什么是政府采购"
            }
        ],
        "web_search": {
            "enable": True,
            "enable_citation": False,
            "enable_trace": False
        }
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-tTXXQUFQTzD0wmpZaZcw8/6339a986fa067a766bb5cb45e94ec619443829d3',
        'appid': 'app-0uIqZTDX'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    
    print(response.text)
    
if __name__ == '__main__':
    main()
