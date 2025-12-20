(ns scicloj.llm.generate-test
  (:require
   [uncomplicate.commons.core :as uc-co-core :refer [with-release]]
   [scicloj.llm.generate :as gen]
   [uncomplicate.diamond.internal.onnxrt.core :refer [clear-bound-outputs bound-values mutable-data value] :as onnxrt-internal]
   )
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]))



(def model-base-dir  "/hf-models")
(def model-name "onnx-community/gemma-3-1b-it-ONNX")
(def tokenizer
  (-> (format "file:%s/%s" model-base-dir model-name)
      java.net.URI.
      java.nio.file.Path/of
      HuggingFaceTokenizer/newInstance))


;; gemma3

; "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a poem about Machine Learning.<end_of_turn>\n<start_of_turn>model\n"
;; (def initial-input-ids [[2    105   2364    107   3048    659    496  11045  16326 236761
;;                          108   6974    786    496  27355   1003  15313  19180 236761    106
;;                          107    105   4368    107]])
(def initial-input-ids
  (-> (.encode tokenizer "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a poem about Machine Learning.<end_of_turn>\n<start_of_turn>model\n")
      .getIds
      vec
      vector))

(def batch-size 1)
(def max-tokens 1)
(.decode tokenizer (long-array (first initial-input-ids)))

(def response
  (with-release [mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)
                 env (onnxrt-internal/environment :warning "test" nil)
                 opt (-> (onnxrt-internal/options)
                         (onnxrt-internal/append-provider! :dnnl))

                 sess (onnxrt-internal/session env (format "%s/%s/onnx/model.onnx" model-base-dir model-name) opt)]

    
    (gen/generate sess mem-info initial-input-ids batch-size
                  (fn [next-token-info]
                    (print (.decode tokenizer (long-array [(:token-id next-token-info)])))
                    (flush))
                  ;;from https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/config.json
                  {:num-key-value-heads 1
                   :num-hidden-layers 26
                   :head-dim 256
                   :vocab-size 262144
                   :eos-token-ids [1 106]}
                  max-tokens)
    (println)))




(comment
  (def mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default))

  (def env (onnxrt-internal/environment :warning "test" nil))

  (def opt (-> (onnxrt-internal/options)
               (onnxrt-internal/append-provider! :dnnl)))

  (def sess (onnxrt-internal/session env (format "%s/%s/onnx/model.onnx" model-base-dir model-name) opt))


  (-> sess onnxrt-internal/output-type-info))



(comment
  (.decode tokenizer (long-array [1106 107 8623 108 659 496 11045 532 236761 107 19058
                                  496 496 17856 1003 506 19180 107 108 249398 19058 70319 9366 19058]))



  (def env (onnxrt-internal/environment :warning "test" nil))

  (def opt
    (-> (onnxrt-internal/options)
        (onnxrt-internal/append-provider! :cuda)))


  (def sess (onnxrt-internal/session env "/hf-models/onnx-community/gemma-3-1b-it-ONNX/onnx/model.onnx" opt)))



