(ns inference-gpt2
  (:require [uncomplicate.neanderthal.core :refer [transfer! iamax native]]
           [uncomplicate.neanderthal.block :refer [buffer]] 
           [uncomplicate.diamond
             [tensor :refer [tensor desc output]]
             [dnn :refer [network]]
             [onnxrt :refer [onnx]]]
           [uncomplicate.diamond.native]
            [uncomplicate.clojure-cpp
             :refer [null? float-pointer long-pointer pointer-vec capacity! put-entry! fill! get-entry
                     pointer-pointer int-pointer pointer-pointer]]
           [uncomplicate.diamond.onnxrt :as onnxrt]
           [uncomplicate.diamond.internal.onnxrt.core :refer :all :as onnxrt-internal]
            [uncomplicate.commons.core :as uc-co-core :refer [info]]
            [uncomplicate.diamond.internal.onnxrt.constants]
            [clojure.tools.build.api :as b]
           ))

(def basis (delay (b/create-basis {:project "deps.edn"})))

(b/javac {:src-dirs ["java"]
           :class-dir "classes"
           :basis @basis
           :javac-opts ["--release" "11"]})

(import GPT2Tokenizer)

(defn predict-onnx [input-ids-array attention-mask-array num-output-tokens]
  
  (uc-co-core/with-release [env (onnxrt-internal/environment nil)
                            opt (onnxrt-internal/options)
                            sess (onnxrt-internal/session env "models/gpt2-lm-head-bs-12.onnx" opt)

                            mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)

                            output-tken-num-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 1]
                                                         (long-pointer [num-output-tokens]))
                            input-ids-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count input-ids-array)]
                                                         (long-pointer input-ids-array))

                            attention-mask-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count attention-mask-array)]
                                                         (float-pointer attention-mask-array))

                            infer! (onnxrt-internal/runner* sess)
                            io-binding (onnxrt-internal/io-binding sess
                                                                   [input-ids-tz attention-mask-tz output-tken-num-tz]
                                                                   [mem-info])]

    (uc-co-core/with-release [outputs (infer! io-binding)]
      (let [output-tensor-shape (-> outputs
                                    bound-values
                                    (get 0)
                                    value-tensor-info
                                    onnxrt-internal/shape)]

        (partition (get output-tensor-shape 1)
                   (-> outputs
                       bound-values
                       (get 0)
                       value
                       mutable-data
                       long-pointer
                       (capacity! (* (get output-tensor-shape 0)
                                     (get output-tensor-shape 1)))
                       (pointer-vec)))))))

(defn ask-gpt [question max-tokens]
  (let [gpt-2-tokenizer (GPT2Tokenizer.)
        input-ids
        (long-array
         (.encode gpt-2-tokenizer question))
        answer (predict-onnx input-ids,
                             (repeat (count input-ids) 1)
                             max-tokens)
        ]
    (.decode gpt-2-tokenizer
             (java.util.Arrays/asList (to-array (int-array (first answer)))))))



(def question "Berlin is the capital of ")
(def answer (ask-gpt question 1))
(println :question question)
(println :answer answer)



