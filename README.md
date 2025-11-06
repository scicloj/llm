# llm

allows to ask questions to GPT2 in pure Clojure

VERY DRAFT !!!

# instructions 
- tries inferencnce on GPT2 and gpt-oss-20
- download https://huggingface.co/onnxmodelzoo/gpt2-lm-head-bs-12 to "scicloj.llm/models/gpt2-lm-head-bs-12.onnx"
- download model.onnx and model.onxx.data from https://huggingface.co/onnxruntime/gpt-oss-20b-onnx/tree/main/cuda/cuda-int4-kquant-block-32-mixed to '/models/gpt-oss-20b/cuda'
(I don't use CDA, but he CPU model doesn't load at all)


- It only works with this model so far.

- I tested it on **CPU** only

- it requires that "deep diamond" works in your setup
   - might need changes in deps.edn

- It is working in the provided **devcontainer** setup 

- current 'deps.edn' probably only works on **Linux + CPU**
  - the devcontainer uses ubuntu (noble), so this 'should' work out of    the box


