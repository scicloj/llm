(ns dd-onnx-oss-gpt-20b
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

(defn predict-onnx [input-ids-array attention-mask-array]
  
  (uc-co-core/with-release [env (onnxrt-internal/environment nil)
        opt (onnxrt-internal/options)
        sess (onnxrt-internal/session env "models/gpt-oss-20b/cpu/model.onnx" opt)



        mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)

        input-ids-tz
        (onnxrt-internal/onnx-tensor mem-info
                                     [1 (count input-ids-array)]
                                     (long-pointer input-ids-array))

        attention-mask-tz
        (onnxrt-internal/onnx-tensor mem-info
                                     [1 (count attention-mask-array)]
                                     (long-pointer attention-mask-array)
                                     )

        all-0-tz (onnxrt-internal/onnx-tensor mem-info
                                              [1 8 1 64]
                                              (float-pointer (repeat 4096 0))
                                              :float16   
                                              )
        infer! (onnxrt-internal/runner* sess)
        io-binding (onnxrt-internal/io-binding sess
                                               (concat
                                                [input-ids-tz attention-mask-tz]
                                                (repeat 48 all-0-tz)
                                                )
                                               [mem-info])]

    (uc-co-core/with-release [outputs (infer! io-binding)
          output-tensor-shape (-> outputs
                                  bound-values
                                  (get 0)
                                  value-tensor-info
                                  onnxrt-internal/shape)]
      
      (-> outputs
          bound-values
          (get 0)
          value
          mutable-data
          float-pointer
          (capacity! (* (get output-tensor-shape 0)
                        (get output-tensor-shape 1)
                        (get output-tensor-shape 2)))
          (pointer-vec))
      )))

(comment 
  (def env (onnxrt-internal/environment nil))
  
  (def opt (onnxrt-internal/options))
  
  (def sess (onnxrt-internal/session env "models/gpt-oss-20b/cpu/model.onnx" opt))

  (session-model-metadata sess)
  (-> sess keys)
  ;;=> "input_ids"
 (-> sess info :input keys)
 ;;=> {:data-type :long, :shape [-1 -1], :address "0xnull"}


  (input-name sess 1)
  ;;=> "attention_mask"
  (input-type-info sess 1)
  ;;=> {:data-type :long, :shape [-1 -1], :address "0xnull"}

  (input-name sess 2)
  ;;=> "past_key_values.0.key"

  (input-type-info sess 2)
  ;;=> {:data-type :float16, :shape [-1 8 -1 64], :address "0xnull"}
  
  (input-name sess 3)
  ;;=> "past_key_values.0.value"
                   
  (input-type-info sess 3)
  ;;=> {:data-type :float16, :shape [-1 8 -1 64], :address "0xnull"}
 
 (-> sess info :output)
  )

;what is ai 
(def what-is-ai {:input-ids [13347 382 20837 1423] :attention_mask [1 1 1 1]})
(def mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default))
;; (def all-0 (onnxrt-internal/onnx-tensor mem-info
;;                                         [1 8 1 64]
;;                                         (long-pointer (repeat 4096 0))))
(println 
 (predict-onnx (:input-ids what-is-ai)
               (repeat (count (:input-ids what-is-ai)) 1)
               ))
                          


(comment
  
  (def question "Berlin is the capital of ")
  
  
  
  (def answer (ask-gpt question 5))
  
  
  
  (println :question question)
  
  
  
  (println :answer answer)
  )
  
