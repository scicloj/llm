(ns inference-oss-gpt-20b
  (:require [uncomplicate.neanderthal.core :refer [transfer! iamax native]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.diamond
             [tensor :refer [tensor desc output]]
             [dnn :refer [network]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.native]
            [uncomplicate.clojure-cpp
             :refer [null? float-pointer long-pointer pointer-vec
                     capacity! put-entry! fill! get-entry
                     pointer-pointer int-pointer pointer-pointer
                     pointer-type byte-pointer short-pointer pointer-seq]]
            [uncomplicate.diamond.onnxrt :as onnxrt]
            [uncomplicate.diamond.internal.onnxrt.core :refer :all :as onnxrt-internal]
            [uncomplicate.commons.core :as uc-co-core :refer [info]]
            [uncomplicate.diamond.internal.onnxrt.constants]
            [libpython-clj2.python :as py]
            [tech.v3.datatype.argops])
  (:import [ai.onnxruntime.platform Fp16Conversions]
           [ai.onnxruntime OrtUtil]))
(py/initialize!)

(def tiktoken (py/import-module "tiktoken"))
(def enc (py/py. tiktoken get_encoding  "o200k_harmony"))



(defn expand-prompt [prompt sess]
  (uc-co-core/with-release [tokens (py/py. enc encode prompt :allowed_special "all")
                            input {:input-ids tokens
                                   :attention_mask (repeat (count tokens) 1)}
                            _ (println :input-tokens tokens)

                            

                            mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)

                            input-ids-array (:input-ids input)
                            attention-mask-array (:attention_mask input)
                            input-ids-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count input-ids-array)]
                                                         (long-pointer input-ids-array))

                            attention-mask-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count attention-mask-array)]
                                                         (long-pointer attention-mask-array))

                            all-0-tz (onnxrt-internal/onnx-tensor mem-info
                                                                  [1 8 1 64]
                                                                  (float-pointer (repeat 4096 0))
                                                                  :float16)
                            infer! (onnxrt-internal/runner* sess)
                            io-binding (onnxrt-internal/io-binding sess
                                                                   (concat
                                                                    [input-ids-tz attention-mask-tz]
                                                                    (repeat 48 all-0-tz))
                                                                   [mem-info])
                            outputs (infer! io-binding)
                            ;; outputs
                            ;; (inference (:input-ids input)
                            ;;            (:attention_mask input))
                            
                            first-bound-value
                            (-> outputs
                                bound-values
                                first
                                value)
                            _ (println :first-bound-value first-bound-value)

                            my-shape
                            (-> first-bound-value
                                info
                                :value
                                :shape)
                            _ (println :my-shape my-shape)


                            my-data--as-short
                            (-> first-bound-value
                                mutable-data
                                short-pointer ;;output tensor has fp16 data , so this 'should' work
                                (capacity! (apply * my-shape))
                                pointer-seq
                                (short-array))
                            _ (println :my-data--as-short my-data--as-short)


                            tensor-data
                            (OrtUtil/reshape my-data--as-short (long-array my-shape))
                            _ (println :tensor-data tensor-data)

                            token-probs
                            (-> tensor-data
                                (get 0)
                                (get (-> my-shape (get 1) dec)))
                            _ (println :token-probs (take 10 token-probs))

                            ;;output tensor has fp16 data , so this 'should' work
                            fp16-token-probs
                            (mapv
                             #(Fp16Conversions/fp16ToFloat %)
                             token-probs)
                            _ (println :fp16-token-probs (take 10 fp16-token-probs))



;(outputs->next-token outputs)
                            next-token
                            (tech.v3.datatype.argops/argmax fp16-token-probs)
                            new-str
                            (py/py. enc decode [next-token])]
    (str prompt new-str)))

(defn expand-and-println [prompt sess]
  (let [new-prompt (expand-prompt prompt sess)]
    (println "----------" new-prompt)
    new-prompt))


(def start-prompt
  "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28
Reasoning: high
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
      <|start|>user<|message|>Which city is larger , Berlin or Rome ?<|end|>
      <|start|>assistant")
;(-> (expand-and-println start-prompt))

(uc-co-core/with-release [env (onnxrt-internal/environment nil)
                          opt (onnxrt-internal/options)
                                                        ; cpu model not working (yet)
                          sess (onnxrt-internal/session env "models/gpt-oss-20b/cuda/model.onnx" opt)]

                          ;; infere next 5 tokens
                          (-> start-prompt
                              (expand-and-println sess)
                              (expand-and-println sess)
                              (expand-and-println sess)
                              (expand-and-println sess)
                              (expand-and-println sess)))




