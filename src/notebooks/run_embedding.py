from api_utils import *
# from dataset_utils import *
# from dataset_utils_extract import *


import json
import sys
import os 
sys.path.append('..')

from utils.dataset_utils_embedding import *
from utils.clinfoAI import ClinfoAIForQA
from config        import OPENAI_API_KEY, NCBI_API_KEY, EMAIL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from tqdm import tqdm
import time
import argparse

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

dataset2processor = {
    'medmcqa': medmcqaProcessor
    ,
    'medqa': medqaProcessor
    ,
    'mmlu': mmluProcessor
    ,
    'qa4mre': qa4mreProcessor
}

def request_api_chatgpt(prompt_CoT):
    GPT_MODEL =  "gpt-3.5-turbo" # 'gpt-4-1106-preview'
    response = client.chat.completions.create(
        messages = [
            {"role": "user", "content" : prompt_CoT}
        ],
        model=GPT_MODEL,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

model2fun = {
    # 'palm': request_api_palm,
    'chatgpt': request_api_chatgpt
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='medmcqa')
    parser.add_argument('--model', type=str, default='3.5')
    parser.add_argument('--notes', type=str, default= '')
    args = parser.parse_args()
    func = ClinfoAIForQA(openai_key=OPENAI_API_KEY, email= EMAIL, engine="Ada", temperature=0.0)
    processor = dataset2processor[args.dataset](func=func)
    data = processor.load_dataset()
    print(len(data))
    generated_data = []
    acc, total_num = 0, 0
    for item in tqdm(data[:100]):
        if args.model == 'palm':
            time.sleep(4)
        prompt = processor.generate_prompt(item)
        if prompt is None:
            continue
        try:
            print('------------retrieving-------')
            # ret = func.pubmed_forward(question=prompt, restriction_date='2020/12/31', num_articles=3, years_back = 10) 
            ret = func.forward(question=prompt, num_articles=3)
            print(ret)
            print('--------finished-------') 
            ret_parsed, acc_item = processor.parse(ret, item)
            print(ret[:2])
            acc += acc_item
            total_num += 1
            if ret_parsed is None:
                continue
            generated_data.append(ret_parsed)
        except Exception as E:
            print('Exception: ',E)
            continue
    # with open(os.path.join('result_{}'.format(args.model), f"{args.dataset}_{args.notes}.json"), 'w+') as f:
    #     json.dump(generated_data, fp=f)

    print(args.dataset, args.model, args.notes)
    print(acc)
    print(total_num)
    print('accuracy:', acc/total_num)

if __name__ == '__main__':
    main()