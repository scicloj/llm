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
   
   ;[uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [fastmath.vector :as v]
   ))

(def resp
  (let [batch-size 1
        past-sequence-length 0
        sequence-length 1
        num-key-value-heads 1
        head-dim 256

        fact (dnnl-factory)
        neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimension! "batch_size" batch-size)
                           (override-dimension! "sequence_length" sequence-length)
                           (override-dimension! "past_sequence_length" past-sequence-length)
                         ;(override-dimension! "past_sequence_length + 256" 1)
                         ;(onnxrt-internal/append-provider! :cuda)
                           ;(override-dimension! "past_sequence_length" 0)
                           ;(override-dimension! "past_sequence_length + 1" 1)
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
                   logits (float-array 262144)
                   ]
      (transfer! [2] input-ids)
      (transfer! [0] position-ids)
      (doseq [pkv past-key-values]
        (transfer! (repeat 0) pkv))
      (transfer! (native (first (smollm-next!))) logits)
      (println  (take 10 (view-vctr (native (first (smollm-next!))))))
      logits
      
      )))


(-> resp first)

;(test-onnx-layer-smollm (cudnn-factory))

