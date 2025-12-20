(ns inference-smollm
  (:require
   [clojure.java.io :as io]
   [uncomplicate.clojure-cpp
    :refer [capacity! float-pointer long-pointer pointer-vec zero!]]
   [uncomplicate.commons.core :refer [with-release]]
   [uncomplicate.diamond.internal.onnxrt.core :refer [append-provider! bound-values environment input-type-info io-binding memory-info mutable-data onnx-tensor options output-type-info override-dimension! runner* session]]
   [fastmath.vector :as v]
   [fastmath.matrix :as mat]
   [uncomplicate.clojurecuda.core])
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]))



(defn step [sess input-ids-seq]

  (with-release [len-input (count input-ids-seq)
                 mem-info (memory-info :cpu :arena 0 :default)
                 input-info (input-type-info sess)
                 output-info (output-type-info sess)
                 input-ids (onnx-tensor mem-info [1 len-input] (long-pointer input-ids-seq))
                 position-ids (onnx-tensor mem-info [1 len-input] (long-pointer (range len-input)))
                 attention-mask (onnx-tensor mem-info [1 len-input] (long-pointer (repeat len-input 1)))
                 past-key-values (vec (repeatedly 60 #(onnx-tensor mem-info [1 3 0 64] (zero! (float-pointer 192)))))
                 present-key-values (vec (repeatedly 60 #(onnx-tensor mem-info [1 3 len-input 64] (zero! (float-pointer (* 3 (inc len-input) 64))))))
                 logits (onnx-tensor mem-info [1 len-input 49152] (float-pointer (* len-input 49152)))
                 data-binding (io-binding sess
                                          {"attention_mask" attention-mask
                                           "input_ids" input-ids
                                           "past_key_values.0.key" (past-key-values 0)
                                           "past_key_values.0.value" (past-key-values 1)
                                           "past_key_values.1.key" (past-key-values 2)
                                           "past_key_values.1.value" (past-key-values 3)
                                           "past_key_values.10.key" (past-key-values 20)
                                           "past_key_values.10.value" (past-key-values 21)
                                           "past_key_values.11.key" (past-key-values 22)
                                           "past_key_values.11.value" (past-key-values 23)
                                           "past_key_values.12.key" (past-key-values 24)
                                           "past_key_values.12.value" (past-key-values 25)
                                           "past_key_values.13.key" (past-key-values 26)
                                           "past_key_values.13.value" (past-key-values 27)
                                           "past_key_values.14.key" (past-key-values 28)
                                           "past_key_values.14.value" (past-key-values 29)
                                           "past_key_values.15.key" (past-key-values 30)
                                           "past_key_values.15.value" (past-key-values 31)
                                           "past_key_values.16.key" (past-key-values 32)
                                           "past_key_values.16.value" (past-key-values 33)
                                           "past_key_values.17.key" (past-key-values 34)
                                           "past_key_values.17.value" (past-key-values 35)
                                           "past_key_values.18.key" (past-key-values 36)
                                           "past_key_values.18.value" (past-key-values 37)
                                           "past_key_values.19.key" (past-key-values 38)
                                           "past_key_values.19.value" (past-key-values 39)
                                           "past_key_values.2.key" (past-key-values 4)
                                           "past_key_values.2.value" (past-key-values 5)
                                           "past_key_values.20.key" (past-key-values 40)
                                           "past_key_values.20.value" (past-key-values 41)
                                           "past_key_values.21.key" (past-key-values 42)
                                           "past_key_values.21.value" (past-key-values 43)
                                           "past_key_values.22.key" (past-key-values 44)
                                           "past_key_values.22.value" (past-key-values 45)
                                           "past_key_values.23.key" (past-key-values 46)
                                           "past_key_values.23.value" (past-key-values 47)
                                           "past_key_values.24.key" (past-key-values 48)
                                           "past_key_values.24.value" (past-key-values 49)
                                           "past_key_values.25.key" (past-key-values 50)
                                           "past_key_values.25.value" (past-key-values 51)
                                           "past_key_values.26.key" (past-key-values 52)
                                           "past_key_values.26.value" (past-key-values 53)
                                           "past_key_values.27.key" (past-key-values 54)
                                           "past_key_values.27.value" (past-key-values 55)
                                           "past_key_values.28.key" (past-key-values 56)
                                           "past_key_values.28.value" (past-key-values 57)
                                           "past_key_values.29.key" (past-key-values 58)
                                           "past_key_values.29.value" (past-key-values 59)
                                           "past_key_values.3.key" (past-key-values 6)
                                           "past_key_values.3.value" (past-key-values 7)
                                           "past_key_values.4.key" (past-key-values 8)
                                           "past_key_values.4.value" (past-key-values 9)
                                           "past_key_values.5.key" (past-key-values 10)
                                           "past_key_values.5.value" (past-key-values 11)
                                           "past_key_values.6.key" (past-key-values 12)
                                           "past_key_values.6.value" (past-key-values 13)
                                           "past_key_values.7.key" (past-key-values 14)
                                           "past_key_values.7.value" (past-key-values 15)
                                           "past_key_values.8.key" (past-key-values 16)
                                           "past_key_values.8.value" (past-key-values 17)
                                           "past_key_values.9.key" (past-key-values 18)
                                           "past_key_values.9.value" (past-key-values 19)
                                           "position_ids" position-ids}
                                          {"logits" logits
                                           "present.0.key" (present-key-values 0)
                                           "present.0.value" (present-key-values 1)
                                           "present.1.key" (present-key-values 2)
                                           "present.1.value" (present-key-values 3)
                                           "present.10.key" (present-key-values 20)
                                           "present.10.value" (present-key-values 21)
                                           "present.11.key" (present-key-values 22)
                                           "present.11.value" (present-key-values 23)
                                           "present.12.key" (present-key-values 24)
                                           "present.12.value" (present-key-values 25)
                                           "present.13.key" (present-key-values 26)
                                           "present.13.value" (present-key-values 27)
                                           "present.14.key" (present-key-values 28)
                                           "present.14.value" (present-key-values 29)
                                           "present.15.key" (present-key-values 30)
                                           "present.15.value" (present-key-values 31)
                                           "present.16.key" (present-key-values 32)
                                           "present.16.value" (present-key-values 33)
                                           "present.17.key" (present-key-values 34)
                                           "present.17.value" (present-key-values 35)
                                           "present.18.key" (present-key-values 36)
                                           "present.18.value" (present-key-values 37)
                                           "present.19.key" (present-key-values 38)
                                           "present.19.value" (present-key-values 39)
                                           "present.2.key" (present-key-values 4)
                                           "present.2.value" (present-key-values 5)
                                           "present.20.key" (present-key-values 40)
                                           "present.20.value" (present-key-values 41)
                                           "present.21.key" (present-key-values 42)
                                           "present.21.value" (present-key-values 43)
                                           "present.22.key" (present-key-values 44)
                                           "present.22.value" (present-key-values 45)
                                           "present.23.key" (present-key-values 46)
                                           "present.23.value" (present-key-values 47)
                                           "present.24.key" (present-key-values 48)
                                           "present.24.value" (present-key-values 49)
                                           "present.25.key" (present-key-values 50)
                                           "present.25.value" (present-key-values 51)
                                           "present.26.key" (present-key-values 52)
                                           "present.26.value" (present-key-values 53)
                                           "present.27.key" (present-key-values 54)
                                           "present.27.value" (present-key-values 55)
                                           "present.28.key" (present-key-values 56)
                                           "present.28.value" (present-key-values 57)
                                           "present.29.key" (present-key-values 58)
                                           "present.29.value" (present-key-values 59)
                                           "present.3.key" (present-key-values 6)
                                           "present.3.value" (present-key-values 7)
                                           "present.4.key" (present-key-values 8)
                                           "present.4.value" (present-key-values 9)
                                           "present.5.key" (present-key-values 10)
                                           "present.5.value" (present-key-values 11)
                                           "present.6.key" (present-key-values 12)
                                           "present.6.value" (present-key-values 13)
                                           "present.7.key" (present-key-values 14)
                                           "present.7.value" (present-key-values 15)
                                           "present.8.key" (present-key-values 16)
                                           "present.8.value" (present-key-values 17)
                                           "present.9.key" (present-key-values 18)
                                           "present.9.value" (present-key-values 19)})
                 next! (runner* sess)]
    (next! data-binding)
    (let [next-bound-values (bound-values data-binding)
          first-bound-value (first next-bound-values)]

      (v/array-vec (pointer-vec (capacity! (float-pointer (mutable-data first-bound-value)) (* len-input 49152)))))))



(def tokenizer (HuggingFaceTokenizer/newInstance (.toPath (io/file "/hf-models/HuggingFaceTB/SmolLM-135M"))))

(defn chat [sess prompt]
  (let [encoding (.encode tokenizer prompt)
        input-ids (seq (.getIds encoding))
        result (step sess input-ids)
        result-mat
        (mat/rows->RealMatrix
         (partition 49152 result))
        result-tokens
        (long-array
         (v/vec->array
          (mat/col
           (->>  result-mat
                 (mat/map-rows v/softmax)
                 (mat/map-rows v/maxdim)) 0)))]

    (.decode tokenizer  (long-array (take-last 1  result-tokens)) true)))

(defn chat-and-callback [sess prompt callback-fn]
  (let [answer (chat sess prompt)
        new-prompt (str prompt answer)]
    (callback-fn prompt)
    new-prompt))

(defn generate [prompt n-tokens callback-fn]
  (with-release
    [env (environment :warning "test" nil)
     opt (-> (options)
             (append-provider! :cuda)
             (override-dimension! "batch_size" 1)
                            ;(override-dimension! "sequence_length"  len-input)
                            ;(override-dimension! "past_sequence_length" len-input)
                            ;(override-dimension! "past_sequence_length + 1" (inc len-input))
                            ;(graph-optimization! :extended)
             )
     sess (session env "/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx" opt)]

    (loop [prompt prompt
           tokens n-tokens]
      (if (pos? tokens)
        (recur (chat-and-callback sess prompt callback-fn)
               (dec tokens))
        prompt))))



(do
  (let [answer (generate "Tell me a story about the hobbits.\n" 100
                         (fn [prompt] (println prompt) (flush)))]
    (println)
    (println "------------------")
    (println answer)))
