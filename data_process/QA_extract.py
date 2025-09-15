import sys
sys.path.append("..")
from infer_llm.opanai_infer import APIInfer
import json
from tqdm import tqdm


prompt = """
# 01 你是一个问答对数据集处理专家。

# 02 你的任务是根据我给出的内容，从里面抽取问答对，每次只生成一个问答对即可。

# 03 答案要全面，请确保答案包含所有必要的信息，充分利用我给出的内容。

# 04 请严格按照我给定的实力模版进行输出，不要添加任何的其他信息，例如json，引号等内容。
    
    {
    "content": "尸检研究发现，抑郁症患者海马、肼胝体膝下区、眶回、背侧前额叶和杏仁核等部位的皮质容量、神经元、胶质细胞数量减少，反复发作抑郁症患者海马容量萎缩。
    应激抑郁模型大鼠海马CA3区神经元增殖减少，椎体细胞树突的数目与长度减少而导致顶树突萎缩。神经病理学研究显示，抑郁症的病因与海马区域神经细胞的萎缩和坏死有关。
    影像学研究表明，抑郁症患者海马体积较正常人明显减小，且缩小的程度与抑郁持续时间呈正相关，从而证实了抑郁症的发病与海马结构关系密切。
    神经元再生在成年哺乳动物海马脑区的减少和增多，分别是导致抑郁症发生和恢复的重要因素。
    应激性海马神经元的损伤和神经元再生障碍共同导致的海马等脑区神经元数量减少是抑郁症发生的关键环节",
    "question": "抑郁症患者的生理症状有哪些。",
    "answer": "抑郁症患者的海马体积明显减小，且萎缩程度与抑郁持续时间呈正相关。海马是负责记忆、情绪调节的重要脑区，其萎缩可能导致认知功能下降和情绪调节障碍，"
    }

# 05 我的内容如下：

"""

# 创建功能
url = "https://api.deepseek.com"               # DeepSeek的API地址
api_key = "sk-9cc79d17dec3438e92ef90db423e20c1"   # DeepSeek的API密钥
model_name = "deepseek-chat"                     # DeepSeek的模型名称
apiinfer = APIInfer(url=url, api_key=api_key, model_name=model_name)    # 创建一个对象

def qa_extract(content):
    messages = [{"role": "user", "content": prompt+content}] # 构建消息列表
    response = apiinfer.infer(messages=messages, stream=False)  # 非流式传输推理结果
    results = response.choices[0].message.content               # 获取回复的文本
    return results

def qa_to_alpaca(qa):
    qa = json.loads(qa)             # 将字符串转换为字典
    qa_new = {
        "instruction": qa["question"],
        "input": "",
        "output": qa["answer"]
    }
    return qa_new


if __name__ == "__main__":
    txt_path = "../data/txt/test.txt"                       # 文本文件的路径
    save_path = "../data/json/test.json"
    qa_total = []                                           # 用于存储所有问答对        

    with open(txt_path, "r", encoding="utf-8") as f:        # 打开文件
        data = f.read().split("\n")                         # 读取文件的内容
    for content in tqdm(data):                                    # 遍历每一行内容
        qa = qa_extract(content)                # 提取问答对
        qa_new = qa_to_alpaca(qa)               # 将问答对转换为Alpaca格式
        qa_total.append(qa_new)                 # 将转换后的问答对添加到列表中

    with open(save_path, "w", encoding="utf-8") as f:  # 将结果保存到文件中
        json.dump(qa_total, f, ensure_ascii=False, indent=4)
    print(qa_total)
