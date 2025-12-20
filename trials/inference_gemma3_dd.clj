(ns
 inference-gemma3-dd
  (:require
   [uncomplicate.commons [core :refer [with-release info]]]
   [uncomplicate.neanderthal.core :refer [iamax transfer! native view-vctr entry!]]
   [uncomplicate.diamond
    [tensor :refer [tensor output *diamond-factory*]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension! append-provider!] :as onnxrt-internal]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [uncomplicate.clojurecuda.core
    :refer [with-context context device cuda-malloc mem-alloc-runtime
            memcpy-host! init stream memset! synchronize!]]
   [fastmath.vector :as v]))
;; LD_LIBRARY_PATH=~/.javacpp/cache/opencl-3.0-1.5.12-linux-x86_64.jar/org/bytedeco/opencl/linux-x86_64

(defn- override-dimensions! [options config]
  (reduce
   (fn [opt [dim value]]
     (override-dimension! opt dim value)
     opt)
   options
   (partition 2
              (:override-dimensions config))))


(defn generate [fact onnx-file input-ids config]
  (def config config)
  (let [;; input-tokens
        ;; [2    105   2364    107   3048    659    496  11045  16326 236761
        ;;  108   6974    786    496  27355   1003  15313  19180 236761    106
        ;;  107    105   4368    107]
        ;batch-size 1
        ;past-sequence-length 1
        sequence-length (count input-ids)
        ;num-key-value-heads 1
        ;head-dim 256
        neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimensions! config))
                   onnx-bp (onnx fact onnx-file {:options opt})
                   input-ids-tz (tensor neand-fact [1 sequence-length] :long :nc)
                   position-ids-tz (tensor neand-fact [1 sequence-length] :long :nc)
                   past-key-values-tzs (repeatedly (:num-hidden-layers config)
                                                   #(tensor fact [(:batch-size config)
                                                                  (:num-key-value-heads config)
                                                                  (:past-sequence-length config)
                                                                  (:head-dim config)] :float :nchw))
                   smollm-next! (onnx-bp (into [input-ids-tz  position-ids-tz] past-key-values-tzs))
                   logits (float-array (* (:vocab-size config) sequence-length))]
      (transfer! input-ids input-ids-tz)
      (transfer! (range sequence-length) position-ids-tz)
      (doseq [pkv past-key-values-tzs]
        (transfer! (repeat 0) pkv))
      (transfer! (native (first (smollm-next!))) logits)
      ;(println  (take 10 (view-vctr (native (first (smollm-next!))))))
      logits)))


(defn infer! [fact model-path input-ids config]
  (try
    (let [next-token
          (->> (generate fact model-path input-ids config)
               (partition (:vocab-size config))
               last
               v/array-vec
               v/maxdim)]
      (println (format "%s : %s " fact next-token)))
    (catch Exception e
      (println (format "infer failed failed for %s : \n %s" fact e)))))

(comment )
(let [input-ids
      [2    105   2364    107   3048    659    496  11045  16326 236761
       108   6974    786    496  27355   1003  15313  19180 236761    106
       107    105   4368    107]
      past-sequence-length 1
      config
      {:batch-size 1
       :past-sequence-length past-sequence-length
       :num-key-value-heads 1
       :head-dim 256
       :num-hidden-layers 52
       :vocab-size 262144
       :override-dimensions
       ["batch_size" 1
        "sequence_length" (count input-ids)
        "past_sequence_length" past-sequence-length]}

      model-path "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx"]
  (infer! (cudnn-factory) model-path input-ids config)
  (infer! (dnnl-factory) model-path input-ids config))










(comment 
  (let [config
        {:batch-size 1
         :past-sequence-length 1
         :num-key-value-heads 3
         :head-dim 64
         :num-hidden-layers 60
         :vocab-size 262144}
        input-ids
        [2]
        model-path "/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx"]
    (infer! (cudnn-factory) model-path input-ids config)
    (infer! (dnnl-factory) model-path input-ids config))
  )
