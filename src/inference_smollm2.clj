(ns
 inference-smollm2
  (:require
   [uncomplicate.commons [core :refer [with-release]]]
   [uncomplicate.fluokitten.core :refer [foldmap]]
   [uncomplicate.neanderthal.core :refer [iamax transfer! native view-vctr entry!]]
   [uncomplicate.diamond
    [tensor :refer [tensor output]]
    [dnn :refer [network activation]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension!]]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [uncomplicate.diamond.internal.neanderthal.factory :refer [vector-factory]]
   ))

(defn test-onnx-layer-smollm [fact]
  (let [neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimension! "batch_size" 1)
                           (override-dimension! "sequence_length" 1)
                           (override-dimension! "past_sequence_length" 1)
                           (override-dimension! "past_sequence_length + 1" 1))
                   src-tz (tensor fact [1 1 28 28] :float :nchw)
                   onnx-bp (onnx fact "/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx" {:options opt})
                   input-ids (tensor neand-fact [1 1] :long :nc)
                   position-ids (tensor neand-fact [1 1] :long :nc)
                   attention-mask (tensor neand-fact [1 1] :long :nc)
                   past-key-values (repeatedly 60 #(tensor fact [1 3 1 64] :float :nchw))
                   smollm-next! (onnx-bp (into [input-ids attention-mask position-ids] past-key-values))]
      (transfer! [2] input-ids)
      (transfer! [0] position-ids)
      (transfer! [1] attention-mask)
      (doseq [pkv past-key-values]
        (transfer! (repeat 0) pkv))
      (println
       (foldmap + 0 -
                (take 10 (view-vctr (native (first (smollm-next!)))))
                [4.141319274902344 -4.2067766189575195 -4.31782341003418 -5.135868072509766 -4.436248779296875
                 -4.15079402923584 2.627662181854248 -4.15079402923584 9.071796417236328 -0.8716740608215332])))))


(test-onnx-layer-smollm (dnnl-factory))
(test-onnx-layer-smollm (cudnn-factory))

