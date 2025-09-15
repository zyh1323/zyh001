from openai import OpenAI


class APIInfer:
    def __init__(self, url, api_key, model_name):
        self.url = url          # 厂商的模型地址
        self.api_key = api_key      # 厂商的模型的apikey,用来身份认证和计费
        self.model_name = model_name         # 厂商的模型的名称
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)       # 建立连接

    def infer(self, messages, stream=True, temperature=1.0, top_p=0.8):
        response = self.client.chat.completions.create(
            model = self.model_name,                 # 使用的模型名称
            messages = messages,                # 消息列表
            stream = stream,                     # 是否开启流式传输
            temperature=temperature,                    # 温度，越大越随机
            top_p = top_p                        # top_p，控制采样的多样性，过滤掉概率较低的词     
        )
        return response


if __name__ == "__main__":
    # 一：创建功能
    url = "https://api.deepseek.com"               # DeepSeek的API地址
    api_key = "sk-9cc79d17dec3438e92ef90db423e20c1"   # DeepSeek的API密钥
    model_name = "deepseek-chat"                     # DeepSeek的模型名称
    apiinfer = APIInfer(url=url, api_key=api_key, model_name=model_name)    # 创建一个对象

    # 二：构建输入
    query = "请介绍一下你自己。"      # 用户输入的消息
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},  # 系统消息，定义助手的角色
        {"role": "user", "content": query}                       # 用户消息，包含用户的输入
    ]           # 构建消息列表

    # 三：调用推理方法
    response = apiinfer.infer(messages=messages)        # 流式传输推理结果
    for res in response:      
        result = res.choices[0].delta.content     # 获取回复的文本
        if result:
            print(result, end="", flush=True)
