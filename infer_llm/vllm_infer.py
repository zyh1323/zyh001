from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VllmInfer():
    def __init__(self, model_path, gpu_memory_utilization=0.8, dtype="bfloat16",
                 tensor_parallel_size=1, max_model_len=1024, max_num_seqs=1):
        self.model = LLM(model=model_path,
                         gpu_memory_utilization=gpu_memory_utilization,     # GPU内存利用率
                         dtype=dtype,                           # 模型数据类型
                         tensor_parallel_size=tensor_parallel_size, # 张量并行大小
                         max_model_len=max_model_len,           # 最大模型长度
                         max_num_seqs=max_num_seqs              # 同时处理的最大请求数
                         )    # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # 加载分词器

    def inference(self, messages, max_tokens=512, temperature=0.6, top_p=0.8, top_k=10, min_p=0.1, 
                  repetition_penalty=1.2):
        inputs = self.tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    tokenize=False)    # 处理输入
        sampling_params = SamplingParams(max_tokens=max_tokens,  # 最大长度
                                        temperature=temperature,  # 温度, 越大，选择小概率的词的可能性越大
                                        top_p=top_p,  # 概率阈值, 0.8表示选择前80%的词
                                        top_k=top_k,  # 选择前k个概率最大的词
                                        min_p=min_p,  # 最小概率
                                        repetition_penalty=repetition_penalty,  # 惩罚重复
                                        )
        outputs = self.model.generate([inputs], sampling_params)[0].outputs[0].text    # 生成输出
        return outputs          # 返回生成的文本


if __name__ == '__main__':
    model_path = "../model/deepseek_1.5B"
    model = VllmInfer(model_path=model_path)      # 初始化模型
    query = "请简要介绍一下你自己。"
    messages = [{"role": "user", "content": query}]  # 构建用户提问的消息格式
    result = model.inference(messages)
    print(result)


# vllm serve /root/autodl-tmp/medai_zyh/model/deepseek-1.5B --port 8080 --api-key abc12345 --served-model-name deepseekchat --gpu_memory_utilization 0.8 --dtype bfloat16 --tensor_parallel_size 1 --max_model_len 1024 --max_num_seqs 1
