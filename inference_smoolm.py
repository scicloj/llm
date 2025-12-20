import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/hf-models/HuggingFaceTB/SmolLM-135M")
tokens = dict(tokenizer("Tell me a story about the hobbits", return_tensors="np"))
options = ort.SessionOptions()
options.add_free_dimension_override_by_name("batch_size",1)
ort_sess = ort.InferenceSession("/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx",
                                sess_options=options)


shape = (1, 3, 0, 64)

tokens["position_ids"]=np.array([range(tokens['input_ids'].shape[1])])

for i in range(30):
    tokens[f"past_key_values.{i}.key"] = np.zeros(shape).astype(np.float32)
    tokens[f"past_key_values.{i}.value"] = np.zeros(shape).astype(np.float32)
    

a=ort_sess.run(["logits"], tokens)
tokenizer.decode(np.argmax(a[0][0],axis=1),)



from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/hf-models/HuggingFaceTB/SmolLM-135M")
inputs = dict(tokenizer("Tell me a story about the hobbits", return_tensors="pt"))
outputs = model.generate(**inputs,max_new_tokens = 100)
print(tokenizer.decode(outputs[0]))

