(ns scicloj.llm.generate
  (:require
   [tech.v3.datatype.argops]
   [uncomplicate.clojure-cpp
    :refer [get-entry long-pointer pointer-pointer pointer-pointer put!
            short-pointer zero! float-pointer capacity! pointer-vec]]
   [uncomplicate.commons.core :as uc-co-core :refer [info with-release]]
   [uncomplicate.diamond.internal.onnxrt.core :refer [clear-bound-outputs bound-values mutable-data value] :as onnxrt-internal]
   [uncomplicate.diamond.native]
   [fastmath.vector :as v])
  )

(defn- ->last-token-id [data-binding sequence_length model-config]
  (let [logits-vec
        (v/array-vec
         (pointer-vec (capacity! (float-pointer (mutable-data (first (bound-values data-binding))))
                                 (* sequence_length (:vocab-size model-config)))))
        logits (partition (:vocab-size model-config) logits-vec)
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

(defn- infer! [sess mem-info  input-ids batch-size model-config]
  (with-release [sequence_length (count (first input-ids))

                 past-key-values
                 (into {}
                       (for [layer (range (:num-hidden-layers model-config))
                             kv ["key" "value"]]
                         [(format "past_key_values.%s.%s" layer kv)
                          (onnxrt-internal/onnx-tensor mem-info [batch-size
                                                                 (:num-key-value-heads model-config)
                                                                 0
                                                                 (:head-dim model-config)]
                                                       (zero! (float-pointer 0)))]))


                 present-key-values
                 (into {}
                       (for [layer (range  (:num-hidden-layers model-config))
                             kv ["key" "value"]]
                         [(format "present.%s.%s" layer kv)
                          (onnxrt-internal/onnx-tensor mem-info [batch-size
                                                                 (:num-key-value-heads model-config)
                                                                 sequence_length
                                                                 (:head-dim model-config)]
                                                       (zero! (float-pointer
                                                               (* batch-size
                                                                  (:num-key-value-heads model-config)
                                                                  sequence_length
                                                                  (:head-dim model-config)))))]))
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
                                  [1 sequence_length (:vocab-size model-config)]
                                  (float-pointer (* sequence_length (:vocab-size model-config)))))


                 data-binding
                 (onnxrt-internal/io-binding sess in-binding out-binding)
                 next! (onnxrt-internal/runner* sess)]


    (next! data-binding)
    (->last-token-id data-binding sequence_length model-config)))



(defn generate [ort-session
                mem-info
                initial-input-ids
                batch-size
                next-token-callback
                model-config
                max-tokens]
  (loop [token-count 0
         accum
         {:input-ids initial-input-ids
          :generated-tokens []}]
    (when (and
           (< token-count max-tokens)
           (not (contains? (into #{} (:eos-token-ids model-config))
                           (last (:generated-tokens accum)))))
      (let [next-token (infer! ort-session mem-info (:input-ids accum)
                               batch-size
                               model-config)]
        (next-token-callback {:token-id next-token})
        (recur
         (inc token-count)
         {:input-ids [(conj (first (:input-ids accum)) next-token)]
          :generated-tokens (conj (:generated-tokens accum) next-token)})))))
