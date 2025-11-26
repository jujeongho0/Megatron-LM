# vLLM

**0. Installation:**
```bash
pip install vllm==0.11.0
pip install transformers==4.57.1
```

**1. Inference with vLLM:**
```bash
import torch
from vllm import ModelRegistry, LLM, SamplingParams

from wbl import WBLForCausalLM

ModelRegistry.register_model("WBLForCausalLM", WBLForCausalLM)

model_path = "/workspace/language-data/lm_team/personal/jeongho/WBL-100B-A10B-S1-HF"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=8,
)

print("")
print(f"Successfully loaded model: {model_path.split('/')[-1]}")

prompts = [
    "대한민국(한국 한자: 大韓民國)은 동아시아의 한반도 군사 분계선 남부에 위치한 나라로,",
    "인공지능(人工智能, 영어: artificial intelligence, AI)은",
    "Charlotte Perriand (24 October 1903 - 27 October 1999) was",
    "Albert Einstein (14 March 1879 - 18 April 1955) was",
    "def fibonacci(n):",
]

sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("")
    print(f"{prompt}{generated_text}")
    print(f"{prompt}{generated_text}")
```
