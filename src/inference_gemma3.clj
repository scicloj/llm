(ns inference-gemma3
  (:require
   [tech.v3.datatype.argops]
   [uncomplicate.clojure-cpp
    :refer [get-entry long-pointer pointer-pointer pointer-pointer
            short-pointer zero! float-pointer]]
   [uncomplicate.commons.core :as uc-co-core :refer [info with-release]]
   [uncomplicate.diamond.internal.onnxrt.constants]
   [uncomplicate.diamond.internal.onnxrt.core :refer [bound-values mutable-data value] :as onnxrt-internal]
   [uncomplicate.diamond.native])
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]
   [ai.onnxruntime.platform Fp16Conversions]))

(def input-ids
  [[2    105   2364    107   3048    659    496  11045  16326 236761
    108   6974    786    496  27355   1003  15313  19180 236761    106
    107    105   4368    107]])


(def attention-mask [[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]])

(def batch-size (count tokens))

(def num-key-value-heads 1)

(def head-dim 256)
(def num-hidden-layers 26)
(def mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default))
(def env (onnxrt-internal/environment :warning "test" nil))
(def opt (-> (onnxrt-internal/options)
             (onnxrt-internal/append-provider! :cuda)
         ;(override-dimension! "batch_size" 1)
                            ;(override-dimension! "sequence_length"  len-input)
                            ;(override-dimension! "past_sequence_length" len-input)
                            ;(override-dimension! "past_sequence_length + 1" (inc len-input))
                            ;(graph-optimization! :extended)
             ))
(def sess (onnxrt-internal/session env "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx" opt))

(def past-key-values
  (into {}
        (for [layer (range (+ 4 num-hidden-layers))
              kv ["key" "value"]]
          [(format "past_key_values.%s.%s" layer kv)
           (onnxrt-internal/onnx-tensor mem-info [batch-size num-key-value-heads 0 head-dim]
                                        (zero! (float-pointer 0)))]

     ;f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32) 
          )))

(def len-input (-> input-ids first count))
(def present-key-values
  (into {}
        (for [layer (range (+ 4 num-hidden-layers))
              kv ["key" "value"]]
          [(format "present.%s.%s" layer kv)
           (onnxrt-internal/onnx-tensor mem-info [batch-size
                                                  num-key-value-heads
                                                  (-> input-ids first count)
                                                  head-dim]
                                        (zero! (float-pointer
                                                (* batch-size
                                                   num-key-value-heads
                                                   len-input
                                                   head-dim))))]

     ;f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32) 
          )))
(def position-ids (range 1 (inc len-input)))
(def in-binding
  (assoc past-key-values
         "attention_mask" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> attention-mask first long-array)))
         "input_ids" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> input-ids first long-array)))
         "position_ids" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> position-ids long-array)))))
(def out-binding
  (assoc present-key-values
         "logits" (onnxrt-internal/onnx-tensor mem-info [1 len-input 262144] (float-pointer (* len-input 262144)))))
(def data-binding
  (onnxrt-internal/io-binding sess in-binding out-binding))

(def next! (onnxrt-internal/runner* sess))

(next! data-binding)

(comment 
  (sort (keys in-binding))
  
  ;;=> ("attention_mask" "input_ids" "past_key_values.0.key" "past_key_values.0.value" "past_key_values.1.key" "past_key_values.1.value" "past_key_values.10.key" "past_key_values.10.value" "past_key_values.11.key" "past_key_values.11.value" "past_key_values.12.key" "past_key_values.12.value" "past_key_values.13.key" "past_key_values.13.value" "past_key_values.14.key" "past_key_values.14.value" "past_key_values.15.key" "past_key_values.15.value" "past_key_values.16.key" "past_key_values.16.value" "past_key_values.17.key" "past_key_values.17.value" "past_key_values.18.key" "past_key_values.18.value" "past_key_values.19.key" "past_key_values.19.value" "past_key_values.2.key" "past_key_values.2.value" "past_key_values.20.key" "past_key_values.20.value" "past_key_values.21.key" "past_key_values.21.value" "past_key_values.22.key" "past_key_values.22.value" "past_key_values.23.key" "past_key_values.23.value" "past_key_values.24.key" "past_key_values.24.value" "past_key_values.25.key" "past_key_values.25.value" "past_key_values.3.key" "past_key_values.3.value" "past_key_values.4.key" "past_key_values.4.value" "past_key_values.5.key" "past_key_values.5.value" "past_key_values.6.key" "past_key_values.6.value" "past_key_values.7.key" "past_key_values.7.value" "past_key_values.8.key" "past_key_values.8.value" "past_key_values.9.key" "past_key_values.9.value" "position_ids")
  
  (sort (onnxrt-internal/input-name sess))
  
  ;;=> ("attention_mask" "input_ids" "past_key_values.0.key" "past_key_values.0.value" "past_key_values.1.key" "past_key_values.1.value" "past_key_values.10.key" "past_key_values.10.value" "past_key_values.11.key" "past_key_values.11.value" "past_key_values.12.key" "past_key_values.12.value" "past_key_values.13.key" "past_key_values.13.value" "past_key_values.14.key" "past_key_values.14.value" "past_key_values.15.key" "past_key_values.15.value" "past_key_values.16.key" "past_key_values.16.value" "past_key_values.17.key" "past_key_values.17.value" "past_key_values.18.key" "past_key_values.18.value" "past_key_values.19.key" "past_key_values.19.value" "past_key_values.2.key" "past_key_values.2.value" "past_key_values.20.key" "past_key_values.20.value" "past_key_values.21.key" "past_key_values.21.value" "past_key_values.22.key" "past_key_values.22.value" "past_key_values.23.key" "past_key_values.23.value" "past_key_values.24.key" "past_key_values.24.value" "past_key_values.25.key" "past_key_values.25.value" "past_key_values.26.key" "past_key_values.26.value" "past_key_values.27.key" "past_key_values.27.value" "past_key_values.28.key" "past_key_values.28.value" "past_key_values.29.key" "past_key_values.29.value" "past_key_values.3.key" "past_key_values.3.value" "past_key_values.4.key" "past_key_values.4.value" "past_key_values.5.key" "past_key_values.5.value" "past_key_values.6.key" "past_key_values.6.value" "past_key_values.7.key" "past_key_values.7.value" "past_key_values.8.key" "past_key_values.8.value" "past_key_values.9.key" "past_key_values.9.value" "position_ids")
  

  (keys out-binding)
  


  (onnxrt-internal/output-name sess)
  )