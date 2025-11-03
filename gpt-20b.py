import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model_name = "openai/gpt-oss-20b"
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer("what is AI ?")


ort_sess = ort.InferenceSession('models/gpt-oss-20b/cpu/model.onnx')

