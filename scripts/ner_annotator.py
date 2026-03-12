from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from together import Together
from transformers import AutoTokenizer
import numpy as np
import os
import pandas as pd
import re
from utils import * 
import keys
import logging
import sys
import pickle
from time import sleep, strftime, localtime

logger = logging.getLogger()
logger.setLevel(logging.INFO)



def estimate_input_tokens(prompt,tokenizer ):
    model_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    token_ids = model_tokenizer.encode(prompt)
    return len(token_ids)


def call_togetherai_api(api_model, lang_factor, user_prompt, system_prompt=None):

    prompt = f"{user_prompt}\n"
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    max_tokens = api_model["max_tokens"] - estimate_input_tokens(system_prompt,api_model["tokenizer"])
    if max_tokens <= 0:
        logger.info(f"ERROR - Prompt too big to process by model")
        return "ERROR - Prompt too big to process by model", None

    # https://github.com/togethercomputer/together-python/blob/main/src/together/resources/chat/completions.py#L16
    params = {
        "model": api_model["entry_point"],
        "messages": messages,
        "temperature": PARAMETERS["temperature"],
        "top_p": PARAMETERS["top_p"],
        "max_tokens": max_tokens,
        "logprobs": 1,
        "stream": False,
        #"seed": PARAMETERS["seed"],#define random
        #"n":2
    }

    retry_flag = False
    while not retry_flag:
        try:
            client_together = Together(api_key=keys.KEYS[api_model["provider"]])
            response = client_together.chat.completions.create(**params)
            retry_flag = True
        except Exception as e:
            print(f"Problem with togetherAI-API: {e}, retrying...")
            if "code: 429" in e._message:
                print("waiting...")
                sleep(1)
            elif "code: 422" in e._message:
                match = re.search(r'must be <= (\d+)\. Given: (\d+) `inputs` tokens and (\d+) `max_new_tokens`', e._message)
                if match:
                    limit = int(match.group(1))
                    inputs = int(match.group(2))
                    if inputs > limit:
                        logger.info(f"ERROR - Prompt (context) too big to process by model")
                        return "ERROR - Prompt (model) too big to process by model", None
                print("Reducing tokens by 100")
                params["max_tokens"] = params["max_tokens"] - 100

            sleep(1)

    content =  response.choices[0].message.content
    tokens_probabilities = [(x[0], x[1], np.round(np.exp(x[1]) * 100, 5)) for x in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.token_logprobs)]

    return content, tokens_probabilities

API_MODELS["Llama3"]["function"] = call_togetherai_api
API_MODELS["Qwen3"]["function"] = call_togetherai_api
API_MODELS["Nemotron"]["function"] = call_togetherai_api
API_MODELS["DeepSeek-V3"]["function"] = call_togetherai_api
API_MODELS["DeepSeek-R1"]["function"] = call_togetherai_api

def read_text_files(data_dir):
    texts = []
    filenames = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
                filenames.append(fname)
    return texts, filenames


def save_xml_result(output, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

def save_json_result(output, output_path):
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with input .txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .xml files")
    parser.add_argument("--model", type=str, default="DeepSeek-R1", choices=list(API_MODELS.keys()), help="Model name")
    parser.add_argument("--prompt_type", type=str, default="fewshot_balise", choices=list(SYSTEM_PROMPT_STYLE.keys()), help="Prompt style")
    parser.add_argument("--lang_factor", type=int, default=3, help="Average characters per token for estimating prompt length")
    return parser.parse_args()


#def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    texts, filenames = read_text_files(args.data_dir)

    api_model = API_MODELS[args.model]
    model_function = api_model["function"]
    system_prompt = SYSTEM_PROMPT_STYLE[args.prompt_type]

    for text, fname in tqdm(zip(texts, filenames), total=len(texts), desc="Processing files"):
        prompt = system_prompt.replace("{text}", text)

        content, _ = model_function(api_model, args.lang_factor, user_prompt="", system_prompt=prompt)

        xml_filename = fname.replace(".txt", ".xml")
        output_path = os.path.join(args.output_dir, xml_filename)

        try:
            #import json
            #json_output = json.loads(content)
            save_xml_result(content, output_path)
        except Exception as e:
            print(f"[WARNING] Failed to parse/save output for {fname}: {e}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)  # fallback to saving raw content

def main():
    """
    Break texts into chunks of 10 lines to avoid truncation of results.
    save raw output and extract only the last <root>...</root> block.
    Merge all extracted contents into one XML file.
    """
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    texts, filenames = read_text_files(args.data_dir)

    api_model = API_MODELS[args.model]
    model_function = api_model["function"]
    system_prompt_template = SYSTEM_PROMPT_STYLE[args.prompt_type]
    chunk_size = 10  # Number of lines (sentences) per chunk

    for text, fname in tqdm(zip(texts, filenames), total=len(texts), desc="Processing files"):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        merged_xml_content = ""

        for chunk_id, chunk_lines in enumerate(chunks):
            chunk_text = "\n".join(chunk_lines)
            prompt = system_prompt_template.replace("{text}", chunk_text)

            content, _ = model_function(api_model, args.lang_factor, user_prompt="", system_prompt=prompt)

            # Save raw output
            chunk_debug_path = os.path.join(args.output_dir, fname.replace(".txt", f".chunk{chunk_id}.raw.xml"))
            with open(chunk_debug_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Extract the last <root>...</root> block only
            #matches = re.findall(r"<root>(.*?)</root>", content, re.DOTALL)
            match = re.search(r'<root>(?!.*<root>)(.*?)</root>', content, re.DOTALL)
            if match:
                inner_content = match.group(1)  # group(1) gets the content inside <root>
                print(inner_content)
                lines = inner_content.strip().splitlines()
                cleaned_lines = [line.strip() for line in lines if line.strip()]

                # Save extracted block separately
                extracted_path = os.path.join(args.output_dir, fname.replace(".txt", f".chunk{chunk_id}.only_root.xml"))
                with open(extracted_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(cleaned_lines))

                # Add to final merge
                merged_xml_content += "\n" + "\n".join(cleaned_lines)


        # Wrap the merged result in a root tag
        final_xml = f"<root>\n{merged_xml_content.strip()}\n</root>"
        
        xml_filename = fname.replace(".txt", ".xml")
        output_path = os.path.join(args.output_dir, xml_filename)
        save_xml_result(final_xml, output_path)

def test_one_example():

    model = "Llama3"

    api_model = API_MODELS[model]
    model_function = API_MODELS[model]["function"]
    #system_prompt = SYSTEM_PROMPT_STYLE["complex_output_format"]
    #system_prompt = SYSTEM_PROMPT_STYLE["complex"]
    system_prompt = SYSTEM_PROMPT_STYLE["simple"]
    #system_prompt = SYSTEM_PROMPT_STYLE["cot"]
    user_prompt = "Giovanni Battista Piranesi (prononcé : [dʒoˈvanni batˈtista piraˈneːzi]), dit Piranèse, né à Mogliano Veneto, près de Trévise, appartenant alors à la république de Venise, le 4 octobre 1720, baptisé le 8 novembre en l'église Saint Moïse à Venise, et mort à Rome le 9 novembre 1778 (à 58 ans), est un graveur et un architecte italien."

    print("USER PROMPT:\n",user_prompt)
    lang_factor = 3

    content, tokens_probabilities = model_function(api_model, lang_factor, user_prompt,
                                                   system_prompt)  # call specific model function

    print("MODEL OUTPUT:\n", content)
    print("TOKEN PROBS:\n", tokens_probabilities)




if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()  # Uncomment to run the main function
    #test_one_example()  # For testing purposes
