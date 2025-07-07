import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("DASHSCOPE_API_KEY"), # 如何获取API
    api_key="sk-bd7ce2e72fd642e4b6df4809a367e5c7", #Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt_template = """## 目标:
你是一名专业的多语言对话数据生成专家，精通中文(zh)、粤语(zh-tw)和英文(en)语言。
## 任务描述:
请你将"id"为:"{$id}"号的文本，"{$query}"，转写为地道的中文和英语和粤语
## 输出格式:返回结果按照json格式返回，字段信息及描述如下:
```json{"id": {"zh":"中文","en": "英文翻译", "zh-tw": "粤语翻译"}}```"""

def send_request(question):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus-0806",  # qwen-plus 属于 qwen3 模型，如需开启思考模式，请参见：https://help.aliyun.com/zh/model-studio/deep-thinking
        messages=[
            {'role': 'user', 'content': question}
        ]
    )
    return completion.choices[0].message.content

def get_batch_result(prompts, save_interval=50, output_path='output.xlsx'):
    res_list = []
    texts_list = []  # 保存对应的原始文本
    batch_count = 0  # 记录批次
    for i in tqdm(range(0, len(prompts))):
        question = prompts[i]
        try:
            res = send_request(question)
        except Exception as e:
            res = str(e)
            print(f"Error processing question {i}: {e}")
        res_list.append(res)
        texts_list.append(question)

        # 每处理save_interval条数据保存一次
        if (i + 1) % save_interval == 0 or (i + 1) == len(prompts):
            batch_count += 1
            # 保存当前批次的数据
            temp_df = pd.DataFrame({'Query': texts_list, 'llm_result': res_list})

            # 如果是第一次保存，创建新文件；否则追加到已有文件
            if batch_count == 1:
                temp_df.to_excel(output_path, index=False)
            else:
                # 读取已有数据并追加新数据
                existing_df = pd.read_excel(output_path)
                updated_df = pd.concat([existing_df, temp_df], ignore_index=True)
                updated_df.to_excel(output_path, index=False)

            # 清空临时列表
            res_list = []
            texts_list = []
            print(f"已保存第 {i + 1} 条数据到 {output_path}")


    return res_list

def run_batch():
    df = pd.read_excel(r'./data/output_langs_all.xlsx', sheet_name='Sheet1')
    # df = df[df['role']=='user']
    df_all_list = df['content'].tolist()[84230:]
    # df_all_list = df['content'].tolist()[18998:20000]
    df_session_id_list = df['session_id'].tolist()[84230:]
    texts = [prompt_template.replace('{$query}', str(text)).replace('{$id}',str(session_id)) for text,session_id in zip(df_all_list,df_session_id_list)]

    # 设置每50条保存一次
    results = get_batch_result(texts, save_interval=2, output_path=r'./data/output_langs_content_84230_.xlsx')
    print("Done!")

if __name__ == '__main__':
    # 单条测试
    # question = prompt_template.replace('{$query}', '你好').replace('{$id}', '12345')
    # response = send_request(question)
    # print(response)

    # # 跑批
    run_batch()