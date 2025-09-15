import sys
sys.path.append("..")
from infer_llm.opanai_infer import APIInfer
from infer_llm.memory_llm import MemoryLLM


# 创建对话功能
url = "https://api.deepseek.com"               # DeepSeek的API地址
api_key = "sk-9cc79d17dec3438e92ef90db423e20c1"                            # DeepSeek的API密钥
model_name = "deepseek-chat"                                            # DeepSeek的模型名称
apiinfer = APIInfer(url=url, api_key=api_key, model_name=model_name)    # 创建一个聊天功能对象
# 创建消息功能
messages = [{"role": "system", "content": "你是一个乐于助人的助手。"}]           # 初始化消息列表
memoryllm = MemoryLLM(messages)                                                # 创建一个消息记忆对象


if __name__ == "__main__":
    while True:
        query = input("\n请提出问题：")   # 用户的输入消息
        if query == "q":
            break               # 输入q退出程序 
        message = {"role": "user", "content": query}          # 构建用户消息
        memoryllm.add_message(message)                      # 添加用户消息到记忆中
        messages = memoryllm.get_messages()                 # 获取当前的消息列表
        print("\n消息记忆：\n", messages)

        # 三：调用推理方法
        result_total = ""                                   # 存储模型回复的文本的全部结果
        response = apiinfer.infer(messages=messages)        # 流式传输推理结果
        for res in response:      
            result = res.choices[0].delta.content          # 获取回复的文本
            if result:
                print(result, end="", flush=True)
                result_total += result                     # 拼接回复的文本的全部结果

        message = {"role": "assistant", "content": result_total}    # 构建助手的消息
        memoryllm.add_message(message)                    # 添加助手的消息到记忆中   
        print("\n", "-"*50)                   # 打印分割线    