import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

transformers.logging.set_verbosity_error()

model_path = "/path/to/wbl_model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("")
print(f"Successfully loaded model: {model_path.split('/')[-1]}")

prompts = [
    "대한민국(한국 한자: 大韓民國)은 동아시아의 한반도 군사 분계선 남부에 위치한 나라로,",
    "인공지능(人工智能, 영어: artificial intelligence, AI)은",
    "Charlotte Perriand (24 October 1903 - 27 October 1999) was",
    "Albert Einstein (14 March 1879 - 18 April 1955) was",
    "def fibonacci(n):",
]

inputs = tokenizer(prompts, padding=True, return_tensors="pt")["input_ids"].to(model.device)
generate_ids = model.generate(inputs, max_new_tokens=128)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
for output in outputs:
    print("")
    print(f"{output!r}")