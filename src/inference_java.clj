(ns inference-java
  (:import [ai.onnxruntime OrtEnvironment]
           [ai.onnxruntime OrtSession$SessionOptions]))

(def env (OrtEnvironment/getEnvironment))


(->
 (.createSession env  "/hf-models/onnx-community/gpt2-ONNX/onnx/model_bnb4.onnx", (OrtSession$SessionOptions.))
 (.getOutputInfo)
 (get "logits")
 (.getInfo))
;;=> #object[ai.onnxruntime.TensorInfo 0x439bb524 
;;"TensorInfo(javaType=FLOAT,onnxType=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,shape=[-1, -1, 50257],dimNames=[batch_size,sequence_length,\"\"])"

(->
 (.createSession env  "/hf-models/onnx-community/gpt2-ONNX/onnx/model_uint8.onnx", (OrtSession$SessionOptions.))
 (.getOutputInfo)
 (get "logits")
 (.getInfo))

(->
 (.createSession env  "/hf-models/onnx-community/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx", (OrtSession$SessionOptions.))
 (.getOutputInfo)
 (get "logits")
 (.getInfo))


