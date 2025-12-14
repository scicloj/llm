(ns
 gemma3-test
  (:require
   [uncomplicate.commons.core :refer [with-release info bytesize size release]]
   [uncomplicate.fluokitten.core :refer [fold fmap!]]
   [uncomplicate.clojure-cpp
    :refer [null? float-pointer long-pointer pointer-vec capacity! put-entry! fill! get-entry
            pointer-pointer zero! get-pointer]]
   [uncomplicate.neanderthal.math :refer [exp]]
   [uncomplicate.diamond.internal.onnxrt.core :refer :all])
  (:import clojure.lang.ExceptionInfo))




(with-release [env (environment :warning "test" nil)
               opt (-> (options)
                       (append-provider! :cuda)
                       (override-dimension! "batch_size" 1)
                       (override-dimension! "sequence_length" 1)
                       (override-dimension! "past_sequence_length" 0)
                       (override-dimension! "total_sequence_length" 1)
                       (graph-optimization! :extended))
               sess (session env "/hf-models/onnx-community/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx" opt)
               mem-info (memory-info :cpu :arena 0 :default)
               input-info (input-type-info sess)
               output-info (output-type-info sess)
               input-ids (onnx-tensor mem-info [1 1] (long-pointer [2]))
               position-ids (onnx-tensor mem-info [1 1] (long-pointer [0]))
               attention-mask (onnx-tensor mem-info [1 1] (long-pointer [1]))
               past-key-values (repeatedly 52 #(onnx-tensor mem-info [1 1 0 256] (float-pointer 0)))
               present-key-values (repeatedly 52 #(onnx-tensor mem-info [1 1 1 256] (float-pointer 256)))
               logits (onnx-tensor mem-info [1 1 262144] (float-pointer 262144))
               data-binding (io-binding sess (into [input-ids attention-mask position-ids] past-key-values)
                                        (into [logits] present-key-values))
               next! (runner* sess)]
  (next! data-binding)
  (println
   (apply +
    (pointer-vec (capacity! (float-pointer (mutable-data (first (bound-values data-binding)))) 262144))))
  (next! data-binding)
  (println
   (apply +
          (pointer-vec (capacity! (float-pointer (mutable-data (first (bound-values data-binding)))) 262144))))
  )