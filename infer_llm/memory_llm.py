class MemoryLLM:
    def __init__(self, messages, max_length=9):
        self.messages = messages                # 存储的消息列表
        self.max_length = max_length            # 最大存储长度

    def add_message(self, message):
        self.messages.append(message)          # 添加新的消息到消息列表
        if len(self.messages) > self.max_length:   # 如果消息列表超过最大长度
            self.messages.pop(1)                # 删除最早的用户消息
            self.messages.pop(1)                # 删除最早的助手消息

    def get_messages(self):
        return self.messages                    # 返回当前的消息列表
    

if __name__ == "__main__":
    query = "请介绍一下你自己。"      # 用户输入的消息
    messgaes = [{"role": "system", "content": "你是一个乐于助人的助手。"},  # 系统消息，定义助手的角色
                {"role": "user", "content": query}]                       # 用户消息，包含用户
    memoryllm = MemoryLLM(messages=messgaes)                # 初始化了一个消息记忆
    print(memoryllm.get_messages())                          # 打印当前的消息列表
    answer = "我是deepseek，一个由深度学习驱动的AI助手。"   # 模拟助手的回复
    messgae = {"role": "assistant", "content": answer}        # 构建助手的消息
    memoryllm.add_message(messgae)                        # 添加助手的消息到记忆中
    print(memoryllm.get_messages())