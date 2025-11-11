(ns inference-gemma
  (:require [uncomplicate.neanderthal.core :refer [transfer! iamax native vctr]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.neanderthal.native :refer [dv]]
            [uncomplicate.diamond
             [tensor :refer [tensor desc output transformer]]
             [dnn :refer [network]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.native]
            [uncomplicate.clojure-cpp
             :refer [null? float-pointer long-pointer pointer-vec
                     capacity! put-entry! fill! get-entry
                     pointer-pointer int-pointer pointer-pointer
                     pointer-type byte-pointer short-pointer pointer-seq]]
            [uncomplicate.diamond.onnxrt :as onnxrt]
            [uncomplicate.diamond.internal.onnxrt.core :refer :all :as onnxrt-internal]
            [uncomplicate.commons.core :as uc-co-core :refer [info]]
            [uncomplicate.diamond.internal.onnxrt.constants]
            [libpython-clj2.python :as py]
            [tech.v3.datatype.argops]
            [clojure.tools.build.api :as b])
  (:import [ai.onnxruntime.platform Fp16Conversions]
           [ai.onnxruntime OrtUtil]))
(py/initialize!)
(py/from-import transformers.models.gemma.tokenization_gemma GemmaTokenizer)

(def tokenizer (GemmaTokenizer :vocab_file "/hf-models/onnx-community/gemma-3-1b-it-ONNX-GQA/tokenizer.model"))

(def tokens
  (py/->jvm (py/py. tokenizer encode "How are you ?")))

(comment

  (def env (onnxrt-internal/environment nil))

  (def opt (onnxrt-internal/options))

                                                      ; cpu model not working (yet)
  (def sess (onnxrt-internal/session env
                                     "/hf-models/onnx-community/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx"
                                     opt))

  (input-name sess)
  ;;=> {:input {"past_key_values.19.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.4.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.16.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.13.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.1.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.2.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.24.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "position_ids" {:data-type :long, :shape [-1 -1]}, 
  ;;   "past_key_values.13.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.6.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.3.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.14.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.18.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.8.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.5.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.14.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.10.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.20.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.6.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.0.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.4.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.21.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.24.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.22.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.11.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.17.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.23.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.25.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.20.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.18.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.22.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.0.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.7.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.9.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.17.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.16.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.12.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.1.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.10.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.12.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "input_ids" {:data-type :long, :shape [-1 -1]}, 
  ;;   "past_key_values.19.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.9.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "attention_mask" {:data-type :long, :shape [-1 -1]}, 
  ;;   "past_key_values.3.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.8.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.11.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.21.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.15.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;   "past_key_values.15.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.2.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.23.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.7.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.25.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;   "past_key_values.5.key" {:data-type :float, :shape [-1 1 -1 256]}}, 

  ;; :output {"present.0.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.1.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.16.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.25.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.2.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.17.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.4.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.16.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.24.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.23.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.19.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.20.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.9.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.14.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.17.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.15.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.3.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.0.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.6.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.3.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.22.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.5.value" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.25.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.20.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.2.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.19.key" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.5.key" {:data-type :float, :shape [-1 1 -1 256]}, 
  ;;          "present.11.value" {:data-type :float, :shape [-1 1 -1 256]},
  ;;          "present.8.value" {:data-type :float, :shape [-1 1 -1 256]},

  (->
   (zipmap
    (input-name sess)
    (input-type-info sess))
   (get "input_ids"))
  ;;=> {:data-type :long, :shape [-1 -1], :address "0xnull"}



  (output-name sess)
  ;;=> ["logits" "present.0.key" "present.0.value" "present.1.key" "present.1.value" "present.2.key" "present.2.value" "present.3.key" "present.3.value" "present.4.key" "present.4.value" "present.5.key" "present.5.value" "present.6.key" "present.6.value" "present.7.key" "present.7.value" "present.8.key" "present.8.value" "present.9.key" "present.9.value" "present.10.key" "present.10.value" "present.11.key" "present.11.value" "present.12.key" "present.12.value" "present.13.key" "present.13.value" "present.14.key" "present.14.value" "present.15.key" "present.15.value" "present.16.key" "present.16.value" "present.17.key" "present.17.value" "present.18.key" "present.18.value" "present.19.key" "present.19.value" "present.20.key" "present.20.value" "present.21.key" "present.21.value" "present.22.key" "present.22.value" "present.23.key" "present.23.value" "present.24.key" "present.24.value" "present.25.key" "present.25.value"]
  (output-type-info sess))

(def env (onnxrt-internal/environment nil))

(def opt (onnxrt-internal/options))

                                                      ; cpu model not working (yet)
(def sess (onnxrt-internal/session env
                                   "/hf-models/onnx-community/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx"
                                   opt))

(def mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default))
(def input-ids-array (long-array tokens))
(def
  input-ids-tz
  (onnxrt-internal/onnx-tensor mem-info
                               [1 (count input-ids-array)]
                               (long-array input-ids-array)))

(def infer! (onnxrt-internal/runner* sess))
(def my-binding (onnxrt-internal/io-binding sess))
(onnxrt-internal/bind-input! my-binding "input_ids" input-ids-tz)

(println (infer! my-binding))
