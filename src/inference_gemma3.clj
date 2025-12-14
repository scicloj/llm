(ns inference-gemma3
  (:require
   [tech.v3.datatype.argops]
   [uncomplicate.clojure-cpp
    :refer [get-entry long-pointer pointer-pointer pointer-pointer put!
            short-pointer zero! float-pointer capacity! pointer-vec]]
   [uncomplicate.commons.core :as uc-co-core :refer [info with-release]]
   [uncomplicate.diamond.internal.onnxrt.core :refer [clear-bound-outputs bound-values mutable-data value] :as onnxrt-internal]
   [uncomplicate.diamond.native]
   [fastmath.vector :as v])
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]))

;(def model-base-dir "/media/xfs/hf-models")


(def num-key-value-heads 1)

(def head-dim 256)
(def num-hidden-layers 26)

(defn- ->last-token-id [data-binding sequence_length]
  (let [logits-vec
        (v/array-vec
         (pointer-vec (capacity! (float-pointer (mutable-data (first (bound-values data-binding))))
                                 (* sequence_length 262144))))
        logits (partition 262144 logits-vec)

        last-logit
        (v/array-vec
         (last logits))]

    ;; (println :logits-vec (take 10 logits-vec))
    ;; (println :first-bound-value (count logits-vec))
    ;; (println first-five)
    ;; (println last-five)

    ;; (assert (v/edelta-eq  (v/array-vec [-17.110178 , -11.298628 ,   1.2429764, -18.418331 ,  -4.9682384])
    ;;                       first-five
    ;;                       0.001)
    ;;         (str "unexpeced first-five: "  first-five))

    ;; (assert (v/edelta-eq  (v/array-vec [-18.370398, -18.326473, -18.303009, -18.506184, -18.182766])
    ;;                       last-five
    ;;                       0.001)
    ;;         (str "unexpeced last-five: "  last-five))

    (v/maxdim last-logit)))

(defn infer! [sess mem-info  input-ids batch-size]
  (with-release [sequence_length (count (first input-ids))

                 past-key-values
                 (into {}
                       (for [layer (range num-hidden-layers)
                             kv ["key" "value"]]
                         [(format "past_key_values.%s.%s" layer kv)
                          (onnxrt-internal/onnx-tensor mem-info [batch-size num-key-value-heads 0 head-dim]
                                                       (zero! (float-pointer 0)))]))


                 present-key-values
                 (into {}
                       (for [layer (range  num-hidden-layers)
                             kv ["key" "value"]]
                         [(format "present.%s.%s" layer kv)
                          (onnxrt-internal/onnx-tensor mem-info [batch-size
                                                                 num-key-value-heads
                                                                 sequence_length
                                                                 head-dim]
                                                       (zero! (float-pointer
                                                               (* batch-size
                                                                  num-key-value-heads
                                                                  sequence_length
                                                                  head-dim))))]))
                 position-ids (range 1 (inc sequence_length))
                 in-binding
                 (assoc past-key-values
                        "input_ids" (onnxrt-internal/onnx-tensor
                                     mem-info
                                     [1 sequence_length]
                                     (long-pointer (-> input-ids first long-array)))
                        "position_ids" (onnxrt-internal/onnx-tensor
                                        mem-info
                                        [1 sequence_length]
                                        (long-pointer (-> position-ids long-array))))


                 out-binding
                 (assoc present-key-values
                        "logits" (onnxrt-internal/onnx-tensor
                                  mem-info
                                  [1 sequence_length 262144]
                                  (float-pointer (* sequence_length 262144))))


                 data-binding
                 (onnxrt-internal/io-binding sess in-binding out-binding)


                 next! (onnxrt-internal/runner* sess)]


    (next! data-binding)


    (->last-token-id data-binding sequence_length)))



(defn generate [ort-session mem-info initial-input-ids batch-size next-token-callback]
  (reduce
   (fn [accum _]
     (let [next-token (infer! ort-session mem-info (:input-ids accum) batch-size)]
       (next-token-callback {:token-id next-token})
       {:input-ids [(conj (first (:input-ids accum)) next-token)]
        :generated-tokens (conj (:generated-tokens accum) next-token)}))
   {:input-ids initial-input-ids
    :generated-tokens []}
   (range 50)))


(def model-base-dir  "/hf-models")
(def model-name "onnx-community/gemma-3-1b-it-ONNX")
(def tokenizer
  (-> (format "file:%s/%s" model-base-dir model-name)
      java.net.URI.
      java.nio.file.Path/of
      HuggingFaceTokenizer/newInstance))


(def initial-input-ids [[2    105   2364    107   3048    659    496  11045  16326 236761
                         108   6974    786    496  27355   1003  15313  19180 236761    106
                         107    105   4368    107]])
(def batch-size 1)

(with-release [mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)
               env (onnxrt-internal/environment :warning "test" nil)
               opt (-> (onnxrt-internal/options)
                       (onnxrt-internal/append-provider! :dnnl))
               sess (onnxrt-internal/session env (format "%s/%s/onnx/model.onnx" model-base-dir model-name) opt)]
  (generate sess mem-info initial-input-ids batch-size
            (fn [next-token-info]
              (print (.decode tokenizer (long-array [(:token-id next-token-info)])))
              (flush))))



;
;; (println
;;  (.decode tokenizer (long-array @generated-tokens)))


(comment
  (.decode tokenizer (long-array [1106 107 8623 108 659 496 11045 532 236761 107 19058
                                  496 496 17856 1003 506 19180 107 108 249398 19058 70319 9366 19058]))



  (def env (onnxrt-internal/environment :warning "test" nil))

  (def opt
    (-> (onnxrt-internal/options)
        (onnxrt-internal/append-provider! :cuda)))


  (def sess (onnxrt-internal/session env "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx" opt)))



