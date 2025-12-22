(ns 
 inference-smolllm-dd
  (:require
   [uncomplicate.commons [core :refer [with-release info]]]
   [uncomplicate.diamond
    [tensor :refer [tensor shape desc]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension!]]
    ]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.neanderthal.core :refer [native transfer! view-vctr]]))

(defn smollm-infer! [fact]
  (let [neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimension! "batch_size" 1)
                           (override-dimension! "sequence_length" 1)
                           (override-dimension! "past_sequence_length" 1)
                           (override-dimension! "past_sequence_length + 1" 1))
 ;                  src-tz (tensor fact [1 1 28 28] :float :nchw)
                   onnx-bp (onnx fact "/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx" {:options opt})
                   input-ids-tz (tensor neand-fact [1 1] :long :nc)
                   position-ids-tz (tensor neand-fact [1 1] :long :nc)
                   attention-mask-tz (tensor neand-fact [1 1] :long :nc)
                   past-key-values-tzs (repeatedly 60 #(tensor fact [1 3 1 64] :float :nchw))
                   inputs-tzs (into [input-ids-tz attention-mask-tz position-ids-tz] past-key-values-tzs)
                   _ (println :inputs-tzs (map desc inputs-tzs))
                   smollm-next! (onnx-bp inputs-tzs)
                   logits (float-array (* 49152 1))
                   ]
      (println :input-ids-tz--info (desc input-ids-tz))
      (println :attention-mask-tz--info (desc attention-mask-tz))
      (println :position-ids-tz--info (desc position-ids-tz))
      (println :past-kv-tz--info (desc (first past-key-values-tzs)))

      (transfer! [2] input-ids-tz)
      (transfer! [0] position-ids-tz)
      (transfer! [1] attention-mask-tz)
      (doseq [pkv past-key-values-tzs]
        (transfer! (repeat 0) pkv))
      (transfer! (native (first (smollm-next!))) logits)
      (take 10 logits)
      
      )))

(with-release [fact (dnnl-factory)]
  (smollm-infer! fact))

(with-release [fact (cudnn-factory)]
  (smollm-infer! fact))