(ns inference-java
  (:import [ai.onnxruntime OrtEnvironment]
           [ai.onnxruntime OrtSession$SessionOptions]
           ))

(def env (OrtEnvironment/getEnvironment)) 

(def session  (.createSession env  "models/gpt-oss-20b/cuda/model.onnx", (OrtSession$SessionOptions.)));

