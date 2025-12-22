import torch
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer,AutoConfig
ort.preload_dlls()
path_to_model = "/hf-models/HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
config = AutoConfig.from_pretrained(path_to_model)

inputs = dict(tokenizer("Tell me a story about the hobbits.", return_tensors="np"))
#{'input_ids': array([[31530,   549,   253,  1977,   563,   260, 11127,  9229,    30]]), 
# 'attention_mask': array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

providers = ["CUDAExecutionProvider"]
decoder_session = ort.InferenceSession(f"{path_to_model}/onnx/model.onnx",
                                       providers = providers)

decoder_session.get_inputs()[3].shape

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers
eos_token_id = 0 # 106 is for <end_of_turn>



## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
input_ids = inputs['input_ids']
position_ids = np.tile(np.arange(1, input_ids.shape[-1] + 1), (batch_size, 1))
attention_mask = np.tile(1,input_ids.shape)


# 3. Generation loop
max_new_tokens = 1
generated_tokens = np.array([[]], dtype=np.int64)
for i in range(max_new_tokens):

  #print("shape past-key_values: " + str(past_key_values["past_key_values.0.key"].shape))
  
  logits, *present_key_values = decoder_session.run(None, dict(
      attention_mask = attention_mask, 
      input_ids=input_ids,
      position_ids=position_ids,
      **past_key_values))

  #print("shape present-key_values: " + str(present_key_values[0].shape))
  #print("shape logits: " + str(logits.shape))

  
  ## Update values for next generation loop
  input_ids = logits[:, -1].argmax(-1, keepdims=True)
  #print("next: " + str(input_ids[0]) + " : " + tokenizer.decode(input_ids[0]))

  position_ids = position_ids[:, -1:] + 1
  attention_mask = attention_mask[:, -1:]
  for j, key in enumerate(past_key_values):
    past_key_values[key] = present_key_values[j]

  generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
  if (input_ids == eos_token_id).all():
    break

  ## (Optional) Streaming
  print(str(input_ids[0]) + " : " +tokenizer.decode(input_ids[0]), end='\n', flush=True)
  #
  
# 4. Output result
#print(tokenizer.batch_decode(generated_tokens))


[x for x in enumerate(past_key_values)]

#print(tokenizer.decode(np.transpose(logits[:].argmax(-1, keepdims=True)[0])[0]))

