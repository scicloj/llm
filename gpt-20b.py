import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import tiktoken





 
#model_name = "openai/gpt-oss-20b"
 
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#test = tokenizer("what is AI ?")

enc = tiktoken.get_encoding("o200k_harmony")
ort_sess = ort.InferenceSession('/hf-models/gpt-oss-20b/cuda/model.onnx')

# print([input.name for input in ort_sess.get_inputs()])
# print([input.shape for input in ort_sess.get_inputs()])
# print([input.type for input in ort_sess.get_inputs()])




def prompts_to_inputs(prompt):
  tokens =  np.atleast_2d(np.array(enc.encode(prompt,allowed_special="all")))
  print("input:  " + str(tokens[0:10]))

  input_names = np.concatenate(
  [
  [ f"past_key_values.{i}.key", f"past_key_values.{i}.value"
    ]    
      for i in range(24)\
  ])

  input_orts = [

  ort.OrtValue.ortvalue_from_numpy(np.zeros([1,8,1,64]).astype(np.float16))  
      for i in range(48)
  ]

  inputs = dict(zip(input_names,input_orts))
  inputs.update({"input_ids": tokens.astype(np.int64),
                "attention_mask": np.atleast_2d(np.ones(len(tokens[0]))).astype(np.int64)
                      })
  return inputs

prompt = """"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28
Reasoning: high
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
      <|start|>user<|message|>Which city is larger , Berlin or Rome ?<|end|>
      <|start|>assistant"""
prompt = "How are you ?"
for i in range(10):
  outputs = ort_sess.run(None, prompts_to_inputs(prompt))
  last_token_dist = outputs[0][0][outputs[0].shape[1]-1]
  #print(np.sum(last_token_dist))
  #print(last_token_dist)
  last_token= last_token_dist.argmax().item()
  
  last_token_string = enc.decode([last_token])
  prompt += last_token_string
  print("output: " +
        str(last_token_dist) + 
        " : " + str(last_token) + 
        " : " + enc.decode([last_token]))
  print(prompt)
  if last_token==200002:
    break;


np.reshape