# Megatron Bridge

**0. Installation:**
```bash
docker run --rm -it -w /workspace -v $(pwd):/workspace \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:25.09

pip install transformers==4.57.1
```

**1. Prepare HF Dummy Model:**
```
Megatron-LM/
├── bridge/
│   ├── WBL-100B-A10B-HF-Dummy/
│   │   ├── config.json # 모델의 Configuration에 따라 수정
│   │   ├── configuration_wbl.py
│   │   ├── generation_config.json
│   │   ├── modeling_wbl.py
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── build_weight_index.py
│   ├── export_megatron_to_hf.py
│   ├── README.md
│   └── wbl_bridge.py
└── ...
```

**2. Build Dummy Model's Weight Index File:**
```bash
python bridge/build_weight_index.py --hf-model bridge/WBL-100B-A10B-HF-Dummy
```

Check `bridge/WBL-20B-A2B-HF-Dummy/model.safetensors.index.json` file.

**3. Export Megatron to HF:**
```bash
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  bridge/export_megatron_to_hf.py \
  \
  --hf-model bridge/WBL-100B-A10B-HF-Dummy \
  --megatron-path /path/to/megatron_model \
  --hf-path bridge/exports/WBL-100B-A10B-HF
```

**4. Generation**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "bridge/exports/WBL-100B-A10B-HF"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Charlotte Perriand (24 October 1903 - 27 October 1999) was"
inputs = tokenizer(prompt, padding=True, return_tensors="pt")["input_ids"].to(model.device)

generate_ids = model.generate(inputs, max_new_tokens=100)
outputs = tokenizer.batch_decode(generate_ids)[0]
print(outputs)
```
