(ns inference-gemma3
  (:require
   [tech.v3.datatype.argops]
   [uncomplicate.clojure-cpp
    :refer [get-entry long-pointer pointer-pointer pointer-pointer
            short-pointer zero! float-pointer capacity! pointer-vec]]
   [uncomplicate.commons.core :as uc-co-core :refer [info with-release]]
   [uncomplicate.diamond.internal.onnxrt.constants]
   [uncomplicate.diamond.internal.onnxrt.core :refer [bound-values mutable-data value] :as onnxrt-internal]
   [uncomplicate.diamond.native]
   [fastmath.vector :as v])
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]
   [ai.onnxruntime.platform Fp16Conversions]))

(def input-ids
  [[2    105   2364    107   3048    659    496  11045  16326 236761
    108   6974    786    496  27355   1003  15313  19180 236761    106
    107    105   4368    107]])


(def attention-mask [[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]])

(def batch-size (count input-ids))

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
        (for [layer (range num-hidden-layers)
              kv ["key" "value"]]
          [(format "past_key_values.%s.%s" layer kv)
           (onnxrt-internal/onnx-tensor mem-info [batch-size num-key-value-heads 0 head-dim]
                                        (zero! (float-pointer 0)))]

     ;f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32) 
          )))

(def len-input (-> input-ids first count))
(def present-key-values
  (into {}
        (for [layer (range  num-hidden-layers)
              kv ["key" "value"]]
          [(format "present.%s.%s" layer kv)
           (onnxrt-internal/onnx-tensor mem-info [batch-size
                                                  num-key-value-heads
                                                  len-input
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
         ;"attention_mask" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> attention-mask first long-array)))
         "input_ids" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> input-ids first long-array)))
         "position_ids" (onnxrt-internal/onnx-tensor mem-info [1 len-input] (long-pointer (-> position-ids long-array)))))
(def out-binding
  (assoc present-key-values
         "logits" (onnxrt-internal/onnx-tensor mem-info [1 len-input 262144] (float-pointer (* len-input 262144)))))


(def data-binding
  (onnxrt-internal/io-binding sess in-binding out-binding))


(def next! (onnxrt-internal/runner* sess))


(println
 :next
 (next! data-binding))



(println
 :wip
 (float-pointer (mutable-data (first (bound-values data-binding)))))

(def logits-vec
  (v/array-vec
   (pointer-vec (capacity! (float-pointer (mutable-data (first (bound-values data-binding))))
                           (* len-input 262144)))))

(println :logits-vec (take 10 logits-vec))

(println
 :first-bound-value
 (count logits-vec))

(def last-logit
  (v/array-vec
   (last
    (partition 262144 logits-vec))))

(def first-five (v/array-vec (take 5 last-logit)))
(println first-five)
(def last-five (v/array-vec (take-last 5 last-logit)))
(println last-five)

(assert (v/edelta-eq  (v/array-vec [-17.110178 , -11.298628 ,   1.2429764, -18.418331 ,  -4.9682384])
                      first-five
                      0.001)
        (str "unexpeced first-five: "  first-five))

(assert (v/edelta-eq  (v/array-vec [-18.370398, -18.326473, -18.303009, -18.506184, -18.182766])
                      last-five
                      0.001)
        (str "unexpeced last-five: "  last-five))

(print :next-token-id (v/maxdim last-logit))
(assert (= 19058 (v/maxdim last-logit)))


