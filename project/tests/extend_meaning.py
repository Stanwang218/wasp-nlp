import os
import sys
from openai import OpenAI
from sympy import content
from tqdm import tqdm
import json

root_dir = os.path.join(os.path.dirname(__file__), '..')
extend_path_dir = os.path.join(root_dir, 'rewrite_column')
sys.path.append(root_dir)

from data import read_carte_classification_datasets


client = OpenAI(
    api_key='xxx',
    base_url="https://api.deepseek.com")

def extend_column_meaning():
    dataset = read_carte_classification_datasets(max_samples=20)
    for ds_name in tqdm(dataset):
        X, y = dataset[ds_name]
        column_name = X.columns.tolist()
        target_name = y.name
        column_str = ','.join(column_name)
        sampled_rows = X.sample(5, random_state=42)
        text_list = []
        for row in sampled_rows.to_dict(orient="records"):
            text = ", ".join([f"{k}={v}" for k, v in row.items()])
            text_list.append(text)
        row_text = "\n".join(text_list)
        
        system_prompt = f"""
                        You are a helpful assistant. \n
                        You are going to help me extend the meaning of the columns from the dataframe given five rows. \n
                        ONLY output the extended meaning of the columns in the following format: \n
                        Column1: extended meaning of column 1 \n
                        Column2: extended meaning of column 2 \n
                        ... \n
                        ColumnN: extended meaning of column N \n
                        """

        input_str = f"""
                    Given columns are '{column_str}' and target is '{target_name}'. \n
                    Here are five rows of the dataframe: \n
                    {row_text} \n
                    Please extend the meaning of the columns based on the given rows and target.\n
                    """         
        print(system_prompt, input_str)
        response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
            "role": "system", 
             "content": f"{system_prompt}"
            },
            {"role": "user", "content": f"{input_str}"},
        ],
        stream=False,
        reasoning_effort="low",
        extra_body={"thinking": {"type": "enabled"}}
        )
        response = response.choices[0].message.content
        print(response)
        with open(os.path.join(extend_path_dir, f'{ds_name}_extended_meaning.txt'), 'w') as f:
            f.write(response)
        # break               

# extend_column_meaning()


def json_output():
    path = '/Users/code/note/wasp_course/dl4nlp-2026/assignment/project/rewrite_column'
    d = {}
    for file_name in os.listdir(path):
        if file_name.endswith('.txt'):
            with open(os.path.join(path, file_name), 'r') as f:
                dataset_name = file_name.replace('_extended_meaning.txt', '')
                tmp_dict = {}
                for line in f:
                    split_line = line.split(':', 1)
                    if len(split_line) == 2:
                        column_name = split_line[0].strip()
                        extended_meaning = split_line[1].strip()
                        tmp_dict[column_name] = extended_meaning
                d[dataset_name] = tmp_dict

    with open(os.path.join(extend_path_dir, 'extended_meanings.json'), 'w') as f:
        json.dump(d, f, indent=4)

json_output()