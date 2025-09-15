from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from peft import PeftModel


class TransformersInfer:
    def __init__(self, model_path , device, dtype):
        self.model_path = model_path      # 模型路径
        self.device = device              # 设备
        self.dtype = dtype                # 模型数据类型
        self.model = AutoModelForCausalLM.from_pretrained(model_path,   # 加载模型
                                                          torch_dtype=dtype  # 模型参数的类型
                                                          ).to(device)
        self.model = PeftModel.from_pretrained(self.model, "../model/train001").to(device)  # 加载LoRA微调模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)      # 加载分词器

    def infer(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages,  # 输入
                                       tokenizer = True,  # 是否使用分词器
                                       add_generation_prompt=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                    ).to(self.device)    # 处理输入
        gen_kwards = {
                        "max_length": 1024,     # 最大生成长度
                        "do_sample": True,      # 开启随机采样
                        "temperature": 0.8,     # 温度，越大选择小概率的词的可能性越高
                        "top_p": 0.8,           # 概率和，选择前概率和为0.9的词
                        "repetition_penalty": 1.2,   # 重复惩罚率
                        "top_k":10,             # 选择前10个概率最高的词
                    }       # 推理参数
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, 
                                skip_prompt=True )   # 跳过输入部分
        thread = Thread(target=self.model.generate,  # 创建线程
                kwargs = {
                    **inputs,    # 输入
                    **gen_kwards,   # 推理参数
                    "streamer": streamer,   # 流式推理器
                })  # 输入的词向量
        thread.start()          # 执行生成字符的这个任务

        return streamer, thread
    
if __name__ == "__main__":
    # 一：创建功能
    model_path = "../model/deepseek_1.5B"   # 模型路径
    device = "cuda"
    dtype = torch.float16  # 模型数据类型
    model = TransformersInfer(model_path=model_path, device=device, dtype=dtype)   # 创建对象

    # 二：构建输入
    query = "你认识张益恒吗？"  # 输入文本
    messages = [{"role": "system", "content": "你是一名乐于助人的助手。"},
                {"role": "user", "content": query}]   # 构建消息列表

    # 三：调用推理方法
    streamer, thread = model.infer(messages=messages)   # 调用推理方法
    for res in streamer:   # 迭代输出
        print(res, end="", flush=True)   # 打印输出
    thread.join()  # 等待线程结束

