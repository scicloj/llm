(ns
 inference-gemma3-dd
  (:require
   [uncomplicate.commons [core :refer [with-release info]]]
   [uncomplicate.fluokitten.core :refer [foldmap]]
   [uncomplicate.neanderthal.core :refer [iamax transfer! native view-vctr entry!]]
   [uncomplicate.diamond
    [tensor :refer [tensor output]]
    [dnn :refer [network activation]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension!] :as onnxrt-internal]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [fastmath.vector :as v]))

(def resp
  (let [input-tokens 
        [2    105   2364    107   3048    659    496  11045  16326 236761
        108   6974    786    496  27355   1003  15313  19180 236761    106
        107    105   4368    107]
        batch-size 1
        past-sequence-length 0
        sequence-length (count input-tokens)
        num-key-value-heads 1
        head-dim 256

        fact (dnnl-factory)
        neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimension! "batch_size" batch-size)
                           (override-dimension! "sequence_length" sequence-length)
                           (override-dimension! "past_sequence_length" past-sequence-length)
                           )
                   onnx-bp (onnx fact "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx" {:options opt})
                   _ (println (-> onnx-bp  info :src))
                   _ (println (-> onnx-bp  info :dst))
                   input-ids (tensor neand-fact [1 sequence-length] :long :nc)
                   position-ids (tensor neand-fact [1 sequence-length] :long :nc)
                   past-key-values (repeatedly 52 #(tensor fact [batch-size 
                                                                 num-key-value-heads 
                                                                 past-sequence-length 
                                                                 head-dim] :float :nchw))
                   smollm-next! (onnx-bp (into [input-ids  position-ids] past-key-values))
                   logits (float-array (* 262144 sequence-length))
                   ]
      (transfer! input-tokens input-ids)
      (transfer! (range sequence-length) position-ids)
      (doseq [pkv past-key-values]
        (transfer! (repeat 0) pkv))
      (transfer! (native (first (smollm-next!))) logits)
      (println  (take 10 (view-vctr (native (first (smollm-next!))))))
      logits
      
      )))

(println 
 (->> resp 
      (partition 262144)
      last
      v/array-vec
      v/maxdim ))

;(test-onnx-layer-smollm (cudnn-factory))

