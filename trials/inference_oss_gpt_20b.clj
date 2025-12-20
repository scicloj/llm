(ns inference-oss-gpt-20b
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
(def basis (delay (b/create-basis {:project "deps.edn"})))

(b/javac {:src-dirs ["java"]
          :class-dir "classes"
          :basis @basis
          :javac-opts ["--release" "11"]})

(import HalfPrecisionFloat)


(py/initialize!)

(def tiktoken (py/import-module "tiktoken"))
(def enc (py/py. tiktoken get_encoding  "o200k_harmony"))

(def np (py/import-module "numpy"))

(defn swap-two [[a b c]]
  [b a c])

(defn expand-prompt [prompt sess]
  ;uc-co-core/with-release
  (uc-co-core/with-release [tokens (py/py. enc encode prompt :allowed_special "all")
                            input {:input-ids tokens
                                   :attention_mask (repeat (count tokens) 1)}
                            _ (println :input-tokens tokens)



                            mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)

                            input-ids-array (:input-ids input)
                            attention-mask-array (:attention_mask input)
                            input-ids-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count input-ids-array)]
                                                         (long-pointer input-ids-array))

                            attention-mask-tz
                            (onnxrt-internal/onnx-tensor mem-info
                                                         [1 (count attention-mask-array)]
                                                         (long-pointer attention-mask-array))

                            all-0-tz (onnxrt-internal/onnx-tensor mem-info
                                                                  [1 8 1 64]
                                                                  (float-pointer (repeat 4096 0))
                                                                  :float16)
                            infer! (onnxrt-internal/runner* sess)
                            io-binding (onnxrt-internal/io-binding sess
                                                                   (concat
                                                                    [input-ids-tz attention-mask-tz]
                                                                    (repeat 48 all-0-tz))
                                                                   [mem-info])
                            outputs (infer! io-binding)
                            ;; outputs
                            ;; (inference (:input-ids input)
                            ;;            (:attention_mask input))
                            
                            first-bound-value
                            (-> outputs
                                bound-values
                                first
                                value)
                            _ (println :first-bound-value first-bound-value)

                            my-shape
                            (-> first-bound-value
                                info
                                :value
                                :shape)
                            _ (println :my-shape my-shape)
                            _ (def my-shape my-shape)


                            my-data--as-short
                            (-> first-bound-value
                                mutable-data
                                short-pointer ;;output tensor has fp16 data , so this 'should' work
                                (capacity! (apply * my-shape))
                                pointer-seq
                                (short-array))
                            _ (println :my-data--as-short (take 10 (seq my-data--as-short)))
                            _ (println :my-data--as-short--count (count my-data--as-short))

                            _ (def my-data--as-short my-data--as-short)
                            



                            ;; tensor-data
                            ;; (->
                            ;;  (py/py. np reshape my-data--as-short my-shape :order "A")
                            ;;  first
                            ;;  last
                            ;;  py/->jvm)
                            
                            tensor-data
                            (OrtUtil/reshape my-data--as-short (long-array  my-shape))
                            _ (println :tensor-data (take 10 tensor-data))
                            _ (println :tensor-data--count (count tensor-data))
                            ;_ (def tensor-data tensor-data)
                            
                            
                            

                            token-probs
                            (-> tensor-data
                                (get 0)
                                (get (-> my-shape (get 1) dec)))
                            _ (println :token-probs (take 10 token-probs))

                            ;;output tensor has fp16 data , so this 'should' work
                            fp16-token-probs
                            (mapv
                             #(.. (HalfPrecisionFloat. (short %))
                                  getFullFloat)
                             token-probs)
                            _ (println :fp16-token-probs (take 10 fp16-token-probs))



                            next-token
                            (tech.v3.datatype.argops/argmax token-probs)
                            _ (println :next-token next-token)
                            new-str
                            (py/py. enc decode [next-token])]

    (println (info outputs))
    (str prompt " " new-str)))

(defn expand-and-println [prompt sess]
  (let [new-prompt (expand-prompt prompt sess)]
    (println "----------" new-prompt)
    new-prompt))


(def start-prompt
  "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28
Reasoning: high
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
      <|start|>user<|message|>Which city is larger , Berlin or Rome ?<|end|>
      <|start|>assistant")
;(-> (expand-and-println start-prompt))

(uc-co-core/with-release [env (onnxrt-internal/environment nil)
                          opt (onnxrt-internal/options)
                                                        ; cpu model not working (yet)
                          sess (onnxrt-internal/session env "models/gpt-oss-20b/cpu/model.onnx" opt)
                          ;outputs (expand-prompt "How are you ?" sess)
                          ]

     ;(println outputs)
  ;; infere next 5 tokens
  (-> "How are you ?"
      (expand-and-println sess)
      ;(expand-and-println sess)
      ;(expand-and-println sess)
      ;(expand-and-println sess)
      ;(expand-and-println sess)
      )

;      
  )


(comment

  (def env (onnxrt-internal/environment nil))

  (def opt (onnxrt-internal/options))

                                                      ; cpu model not working (yet)
  (def sess (onnxrt-internal/session env
                                     "models/gpt-oss-20b/cpu/model.onnx"
                                     opt))


  (output-name sess))

(comment


  (->
   (py/py. np reshape [1 2 3 4 5 6 7 8] [1 2 4])
   first
   last)


  (->
   (py/py. np reshape [1 2 3 4 5 6] [1 1 6])
   first
   last
   py/->jvm))
(require 'tech.v3.tensor.dimensions)
(tech.v3.tensor.dimensions/create-dimension-transforms
 [[1 100 4] [1 100 4]])



(def tz-1 (tensor {:shape [1 2 10] :data-type :float :layout :ntc}))
(def tz-2 (tensor {:shape [1 2 10] :data-type :float :layout :tnc}))

(def tf (transformer tz-1 tz-2))

(transfer! (range (* 2 10)) tz-1)
(transfer! (range (* 2 10)) tz-2)

(seq (native tz-1))
;;=> {:shape [1 2 10], :data-type :float, :layout [10 10 1]} (0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0)
(seq (native tz-2))
;;=> {:shape [1 2 10], :data-type :float, :layout [20 10 1]} (0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0)


(transfer! (range 1 15) tz-x)
(transfer! (range 1 15) tz-x-ntc)
(seq (native tz-x))
;;=> (1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0)

(seq (native tz-x-ntc))
;;=> (1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0)

(transfer! tz-x-ntc tz-x)

(seq (native tz-x))
;;=> (1.0 3.0 5.0 7.0 9.0 11.0 13.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0)


my-data--as-short
(def tz-x-ntc (tensor my-shape :uint16 :ntc))
(def tz-x (tensor  [2 7 1] :float :tnc))

