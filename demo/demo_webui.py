import sys
sys.path.append("..")  # 将上级目录添加到系统路径中
import gradio as gr
import time
from infer_llm.opanai_infer import APIInfer
from infer_llm.memory_llm import MemoryLLM


# 一：创建功能
url = "https://api.deepseek.com"               # DeepSeek的API地址
api_key = "sk-9cc79d17dec3438e92ef90db423e20c1"   # DeepSeek的API密钥
model_name = "deepseek-chat"                     # DeepSeek的模型名称
apiinfer = APIInfer(url=url, api_key=api_key, model_name=model_name)           # 创建一个对话功能对象
# 创建消息功能
messages = [{"role": "system", "content": "你是一个乐于助人的助手。"}]           # 初始化消息列表
memoryllm = MemoryLLM(messages)                                                # 创建一个消息记忆对象


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")                      # 聊天窗口
    msg = gr.Textbox()                                         # 输入框
    clear = gr.Button("Clear")                                 # 清除按钮
    def user(user_message, history: list):                     # 用户输入的处理逻辑
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):                                    # 机器人的回复逻辑
        query = history[-1]["content"]                         # 取出用户的最新输入
        message = {"role": "user", "content": query}           # 将query构建为消息
        memoryllm.add_message(message)                         # 添加用户消息到记忆中  
        messages = memoryllm.get_messages()                    # 获取当前的消息列表
        bot_message = apiinfer.infer(messages=messages)        # 流式传输推理结果
        result_total = ""                                      # 存储模型回复的文本的全部结果
        history.append({"role": "assistant", "content": ""})
        for character in bot_message:                          # 遍历结果
            character = character.choices[0].delta.content     # 遍历这个生成器，取到其中的每一个字符。
            result_total += character                          # 拼接回复的文本的全部结果
            history[-1]['content'] += character
            time.sleep(0.02)
            yield history
        message = {"role": "assistant", "content": result_total}    # 构建助手的消息
        memoryllm.add_message(message)                         # 添加助手的消息到记忆中   

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8039, share=True)  # 启动服务，开启公网访问