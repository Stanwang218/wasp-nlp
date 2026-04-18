from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def topk_sampling(model, prompt, max_len, topk, tokenizer, temperature=1.0, device="cuda:0"):
    model.eval()
    model.to(device)
    input_ids = torch.tensor(tokenizer(prompt, return_tensors='pt')['input_ids'])[:, :-1] # remove the <EOS> token
    input_ids = input_ids.to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_len - input_ids.shape[1]):
            outputs = model(generated)
            if isinstance(outputs, CausalLMOutputWithPast):
                outputs = outputs.logits
            next_token_logits = outputs[:, -1, :]
            topk_logits, topk_indices = torch.topk(next_token_logits, topk)
            probabilities = torch.softmax(topk_logits / temperature, dim=-1)
            next_token = topk_indices[0, torch.multinomial(probabilities, num_samples=1)[0]]
            # print(next_token)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    # text = " ".join([tokenizer.convert_ids_to_tokens(idx.item()) for idx in generated[0]])
    text = tokenizer.decode([idx.item() for idx in generated[0]])
    return text

login("xxx")


model_name = "allenai/OLMo-2-0425-1B"
save_path = "/mimer/NOBACKUP/groups/oovgen/ziyuan/wasp-nlp/ckpt"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_path)
print(topk_sampling(model, ["In natural language processing, a Transformer"], max_len=50, topk=10, tokenizer=tokenizer, temperature=2.0))
print(topk_sampling(model, ["Is Stockholm the capital of Sweden? Answer yes or no. The answer is"], max_len=50, topk=10, tokenizer=tokenizer, temperature=2.0))
print(topk_sampling(model, ["Write a Python program that reverses a list."], max_len=50, topk=10, tokenizer=tokenizer, temperature=2.0))
