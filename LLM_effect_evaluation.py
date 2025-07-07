'''
功能：LLM回复质量评估
输入：输入文件为 parsed_output_all_datas.xlsx，包含2列：id 和 content
输出：输出文件为 output_effect_evaluation.xlsx，包含4列：query, llm_answer, query_effect_evalutation, result_effect_evaluation
'''

import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("DASHSCOPE_API_KEY"), # 如何获取API
    api_key="sk-bd7ce2e72fd642e4b6df4809a367e5c7",
    # Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt_template1 = """
## 目标:
你是一名专业的多语言对话数据生成专家，精通中文(zh)、粤语(zh-tw)和英文(en)语言。
## 任务描述:
请你将"id"为:"{$id}"号的文本，"{$content}"，转写为地道的中文和英语和粤语
## 输出格式:返回结果按照json格式返回，字段信息及描述如下:
```json{"id": {"zh":"中文","en": "英文翻译", "zh-tw": "粤语翻译"}}```
"""

prompt_template2 = """
你是一个严格的知识回复专家，请根据以下维度和指标评估每个大模型的知识回复答案的质量：
【评测维度】
*事实准确性：1.回复内容是否与公认事实、提供的知识信息一致;2.是否包含已被证伪的错误信息;3.数字、日期、统计数据的精确性
*是否为有效回答：1.是否精确解答了用户的问题;2.是否遗漏知识中的关键信息;3.对复杂问题是否有系统性回答;4.是否将用户全部问题都有回答
*时效性验证：1.提供的信息是否根据提供知识的最新版本;2.对时效敏感问题是否标注时间范围
*逻辑完整性：1.针对知识中无法解答的问题是否有合理的拒答,不能脱离知识进行捏造

【逐步验证】
请按CoT步骤验证思考过程：
Step1 核查事实准确性：数字类,推理计算类需要判断
Step2 完整性检查：是否有效回答
Step3 时效性验证：Query与知识时效保持一致
Step4 逻辑完整性：是否有合理的拒答

【最点终判断】
▶ 合格输出：[PASS] 通过所有检查项
▶ 存在#N/A答案：[NA] 大模型Result生成失败
▶ 不合格：[FAIL] 违反上述四
请用返回结果：PASS/NA/FAIL

【需要评估的问题及回复如下】
用户提问：{$query}
模型回复：{$llm_answer}

【输出格式】
```json{"result": "PASS/NA/FAIL", "reason": "评估理由"} ```
"""


def send_request(question):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus-0806",
        # qwen-plus 属于 qwen3 模型，如需开启思考模式，请参见：https://help.aliyun.com/zh/model-studio/deep-thinking
        messages=[
            {'role': 'user', 'content': question}
        ]
    )
    return completion.choices[0].message.content


def get_batch_result(querys, save_interval=20, output_path='output.xlsx'):
    list_query = []  # 保存已处理的原始query
    list_llm_answer = [] # 保存大模型的回答
    list_query_effect_evalutation = []  # 保存llm_answer效果评估的query
    list_result_effect_evaluation = []  # 保存llm_answer效果评估的结果
    batch_count = 0  # 记录批次
    for i in tqdm(range(0, len(querys))):
        # 调用LLM，生成回复
        query = querys[i]
        try:
            ans = send_request(query)
        except Exception as e:
            ans = str(e)
            print(f"Error processing question {i}: {e}")
        list_query.append(query)
        list_llm_answer.append(ans)
        # 再次调用LLM，评估回复质量
        query_effect_evalutation = prompt_template2.replace('{$query}', query).replace('{$llm_answer}', ans)
        try:
            result_effect_evaluation = send_request(query_effect_evalutation)
        except Exception as e:
            result_effect_evaluation = str(e)
            print(f"Error processing evaluation for question {i}: {e}")
        list_query_effect_evalutation.append(query_effect_evalutation)
        list_result_effect_evaluation.append(result_effect_evaluation)

        # 每处理save_interval条数据保存一次
        if (i + 1) % save_interval == 0 or (i + 1) == len(querys):
            batch_count += 1
            # 保存当前批次的数据
            temp_df = pd.DataFrame({'query': list_query, 'llm_answer': list_llm_answer, 'query_effect_evalutation': list_query_effect_evalutation, 'result_effect_evaluation': list_result_effect_evaluation})

            # 如果是第一次保存，创建新文件；否则追加到已有文件
            if batch_count == 1:
                temp_df.to_excel(output_path, index=False)
            else:
                # 读取已有数据并追加新数据
                existing_df = pd.read_excel(output_path)
                updated_df = pd.concat([existing_df, temp_df], ignore_index=True)
                updated_df.to_excel(output_path, index=False)

            # 清空临时列表
            list_query = []
            list_llm_answer = []
            list_query_effect_evalutation = []
            list_result_effect_evaluation = []
            print(f"已保存第 {i + 1} 条数据到 {output_path}")

    return list_llm_answer


def run_batch(input_path=r'./data/parsed_output_all_datas.xlsx', output_path=r'./data/output_effect_evaluation.xlsx'):
    df = pd.read_excel(input_path, sheet_name='Sheet1')
    # 提取'id'和'content'列
    id_list = df['id'].tolist()
    content_list = df['content'].tolist()
    # 生成query列表
    querys = [prompt_template1.replace('{$content}', str(content)).replace('{$id}', str(idd)) for content, idd in
              zip(content_list, id_list)]
    # 调用大模型，设置每2条保存一次
    results = get_batch_result(querys[:10], save_interval=2, output_path=output_path)
    print("Done!")


if __name__ == '__main__':
    # 单条测试
    # query = prompt_template1.replace('{$content}', '你好').replace('{$id}', '12345')
    # response = send_request(query)
    # print(response)

    # 跑批
    import argparse
    parser = argparse.ArgumentParser(description='LLM Effect Evaluation Script')
    parser.add_argument('--input_path', type=str, default='./data/parsed_output_all_datas.xlsx', help='Path to the input Excel file')
    parser.add_argument('--output_path', type=str, default='./data/output_effect_evaluation.xlsx', help='Path to the output Excel file')
    args = parser.parse_args()

    run_batch(input_path=args.input_path, output_path=args.output_path)

