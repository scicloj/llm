(ns
 inference-gemma3-dd
  (:require
   [uncomplicate.commons [core :refer [with-release info]]]
   [uncomplicate.diamond
    [tensor :refer [tensor shape desc]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension!]]]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.neanderthal.core :refer [native transfer! view-vctr]]
   [fastmath.vector :as v]))
;; need to find "somehow" the OpenCL native libraries
;; ex:
;; LD_LIBRARY_PATH=~/.javacpp/cache/opencl-3.0-1.5.12-linux-x86_64.jar/org/bytedeco/opencl/linux-x86_64

(defn- override-dimensions! [options config]
  (reduce
   (fn [opt [dim value]]
     (override-dimension! opt dim value)
     opt)
   options
   (partition 2
              (:override-dimensions config))))


(defn generate [fact onnx-file input-ids config session-opts]
  (let [sequence-length (count input-ids)
        neand-fact (neanderthal-factory fact)]
    (with-release [onnx-bp (onnx fact onnx-file {:options session-opts})
                   ;_ (println :onnx-bp--info (info onnx-bp))

                   input-ids-tz (tensor neand-fact [1 sequence-length] :long :nc)
                   ;_ (println :input-ids-tz--info (desc input-ids-tz))
                   position-ids-tz (tensor neand-fact [1 sequence-length] :long :nc)
                   ;_ (println :position-ids-tz--info (desc position-ids-tz))
                   attention-mask-tz (tensor neand-fact [1 sequence-length] :long :nc)
                   ;_ (println :attention-mask-tz--info (desc attention-mask-tz))
                   past-key-values-tzs (repeatedly (* 2 (:num-hidden-layers config))
                                                   #(tensor fact [(:batch-size config)
                                                                  (:num-key-value-heads config)
                                                                  (:past-sequence-length config)
                                                                  (:head-dim config)]
                                                            :float :nchw))
                   _ (println :past-kv-tz--info (desc (first past-key-values-tzs)))

                   inputs-tzs (if (:use-attention-mask? config)
                                (into [input-ids-tz attention-mask-tz position-ids-tz] past-key-values-tzs)
                                (into [input-ids-tz position-ids-tz] past-key-values-tzs))
                   ;_ (println :inputs-tzs (map desc inputs-tzs))
                   smollm-next! (onnx-bp inputs-tzs)
                   ;_ (println :smollm-next! smollm-next!)
                   logits (float-array (* (:vocab-size config) sequence-length))]


      (transfer! (vec input-ids) input-ids-tz)
      ;(println :transfered-1)
      (transfer! (vec (range sequence-length)) position-ids-tz)
      ;(println :transfered-2)
      (when (:use-attention-mask? config)
        (transfer! (vec (repeat sequence-length 1)) attention-mask-tz))
      ;(println :transfered-3)
      (doseq [pkv past-key-values-tzs]
        (transfer! (repeat 0) pkv))
      ;(println :transfered-4)
      (transfer! (native (first (smollm-next!))) logits)
      ;(println :back-transfered)
      ;(println  (take 10 (view-vctr (native (first (smollm-next!))))))
      logits)))


(defn infer! [fact model-path input-ids config session-opts]
  (try
    (let [next-token
          (->> (generate fact model-path input-ids config session-opts)
               (partition (:vocab-size config))
               last
               v/array-vec
               v/maxdim)]
      (println (format "%s : %s " fact next-token)))
    (catch Exception e
      (println e)
      (println (format "infer failed failed for %s : \n %s" fact e)))))

(let [input-ids
      [2    105   2364    107   3048    659    496  11045  16326 236761
       108   6974    786    496  27355   1003  15313  19180 236761    106
       107    105   4368    107]
      ; next is 19508
      past-sequence-length 1
      config
      {:batch-size 1
       :past-sequence-length past-sequence-length
       :num-key-value-heads 1
       :head-dim 256
       :num-hidden-layers 26
       :vocab-size 262144
       :use-attention-mask? false}
      session-opts (-> (options)
                       (override-dimension! "batch_size" 1)
                       (override-dimension! "sequence_length" (count input-ids))
                       (override-dimension! "past_sequence_length" past-sequence-length))

      
      model-path "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx"]

  (infer! (cudnn-factory) model-path input-ids config session-opts)
  (infer! (dnnl-factory) model-path input-ids config session-opts))


(let [input-ids [31530]
      len-input (count input-ids)

      config
      {:batch-size 1
       :past-sequence-length 1
       :num-key-value-heads 3
       :head-dim 64
       :num-hidden-layers 30
       :vocab-size 49152
       :use-attention-mask? true
       }
      model-path "/hf-models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx"
      session-opts (-> (options)
                       (override-dimension! "batch_size" 1)
                       (override-dimension! "sequence_length" (count input-ids))
                       (override-dimension! "past_sequence_length" 1)
                       (override-dimension! "past_sequence_length + 1" 1))]

  (infer! (cudnn-factory) model-path input-ids config session-opts)
  (infer! (dnnl-factory) model-path input-ids config session-opts))





; LD_LIBRARY_PATH=/home/vscode/.javacpp/cache/opencl-3.0-1.5.12-linux-x86_64.jar/org/bytedeco/opencl/linux-x86_64/  java -jar /home/vscode/.vscode-server/extensions/betterthantomorrow.calva-2.0.542/deps.clj.jar -Sdeps '{:deps {nrepl/nrepl {:mvn/version,"1.5.1"},cider/cider-nrepl {:mvn/version,"0.58.0"}}}' -M:linux -m nrepl.cmdline --middleware "[cider.nrepl/cider-middleware]"