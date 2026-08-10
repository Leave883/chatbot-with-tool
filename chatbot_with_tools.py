import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
if not API_KEY or not BASE_URL:
    raise RuntimeError("请先设置环境变量 API_KEY 和 BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 用 JSON 格式描述模型可以使用的工具，不是工具本身，而是告知模型的工具说明书。Tool Calling 的参数世界基本就是 JSON + JSON Schema。
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"] # reqiured表示必填的参数
            }
        }
    }
        
]

# 工具的真正实现（实际项目中这里会调用天气 API）
def get_weather(city):
    weather_data = {
        "北京": {"temperature": 8, "condition": "多云"},
        "上海": {"temperature": 15, "condition": "晴"},
        "广州": {"temperature": 22, "condition": "阵雨"},
    }
    data = weather_data.get(city, {"temperature": "未知", "condition": "未知"})
    # json.dumps 把python字典转成 JSON 字符串，因为要返回给模型。ensure_ascii=False 让中文正常显示
    return json.dumps(data, ensure_ascii=False)


# 工具名到函数的映射，因为模型只会返回请求调用，python程序需要自己找到get_weather函数
tool_functions = {"get_weather": get_weather}

SYSTEM_PROMPT = "你是一个友好的助手，可以查询天气。请记住用户告诉你的信息。"
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"[人设] {SYSTEM_PROMPT}")
print("输入消息开始聊天，输入 q 退出\n")

while True:
    user_input = input("你: ")
    if user_input.strip() == "q":
        break

    messages.append({"role": "user", "content": user_input})

    # 把工具列表传给 API，模型会自己判断是否需要调用，第一次调用
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=tools, # 比原来的多上传了工具列表
        extra_body={"thinking": {"type": "disabled"}}
    )
    assistant_message = response.choices[0].message

    # 如果模型决定调用工具，必须多这一步，不然模型第一步会返回需要调用get_weather，而不是回答
    if assistant_message.tool_calls: # 若是前面（行68）上传了tools而且模型判断需要使用tools时，才产生tool_calls
        messages.append(assistant_message) # 把刚刚返回的tool call message加入上下文
        for tool_call in assistant_message.tool_calls:
            # arguments 是 JSON 字符串，需要解析成字典
            args = json.loads(tool_call.function.arguments)
            # 此时tool_call.function.name = "get_weather"，func = tool_functions["get_weather"] = get_weather
            func = tool_functions[tool_call.function.name] 
            # **args 把字典解包成关键字参数，等价于 func(city="北京")
            result = func(**args)
            print(f"  [调用工具] {tool_call.function.name}({args}) => {result}")
            # role 为 "tool" 表示这是工具返回的结果
            # tool_call_id 用来关联这条结果对应哪个工具调用
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        # 拿到工具结果后再调一次模型，让它生成自然语言回答，第二次调用
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}}
        )
        assistant_message = response.choices[0].message

    messages.append({"role": "assistant", "content": assistant_message.content})
    print(f"AI: {assistant_message.content}\n")