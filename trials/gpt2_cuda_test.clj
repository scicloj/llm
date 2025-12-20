(ns
 gpt2_cuda_test
  (:require
   [uncomplicate.commons.core :refer [with-release info bytesize size release]]
   [uncomplicate.fluokitten.core :refer [fold fmap!]]
   [uncomplicate.clojure-cpp
    :refer [null? float-pointer long-pointer pointer-vec capacity! put-entry! fill! get-entry
            pointer-pointer pointer]]
   [uncomplicate.clojurecuda.core
    :refer [with-context context device cuda-malloc mem-alloc-runtime
            memcpy-host! init stream memset! synchronize!]]
   [uncomplicate.neanderthal.math :refer [exp]]
   [uncomplicate.diamond.internal.onnxrt.core :refer :all])
  (:import clojure.lang.ExceptionInfo))

(uncomplicate.diamond.internal.onnxrt.core/init-ort-api!)
(uncomplicate.clojurecuda.core/init)
(filter #{:cuda :dnnl :cpu} (available-providers))


(with-release [dev (device 0)]
  (with-context (context dev :map-host)

    (with-release [env (environment :warning "test" nil)
                   ;hstream (stream)
                   opt (-> (options)
                           (append-provider! :cuda ;{:stream hstream}
                                             )
                           (override-dimension! "batch_size" 1) ;;optional
                           (override-dimension! "seq_len" 3) ;;optional
                           (graph-optimization! :extended))
                   sess (session env "/hf-models/onnxmodelzoo/gpt2-lm-head-bs-12/gpt2-lm-head-bs-12.onnx" opt)
                   cpu-mem-info (memory-info :cpu :arena 0 :default)
                   cuda-mem-info (memory-info :cuda :device 0 :default)
                   input-ids-data (mem-alloc-runtime (* 3 Long/BYTES) :long)
                   input-ids (onnx-tensor cuda-mem-info [1 3] input-ids-data) ;; Grass is
                   attention-mask-data (memset! (mem-alloc-runtime (* 3 Float/BYTES) :float) (float 1.0))
                   attention-mask (onnx-tensor cuda-mem-info [1 3] attention-mask-data)
                   out-token-num-data (mem-alloc-runtime Long/BYTES :long)
                   out-token-num (onnx-tensor cuda-mem-info [1] out-token-num-data)
                   lp-v0-2866-data (mem-alloc-runtime (* 4 36 Long/BYTES) :long)
                   lp-v0-2866 (onnx-tensor cpu-mem-info [4 36] lp-v0-2866-data);; The ORT itself complains about the shape whatever I put. Only mem-info works
                   data-binding (io-binding sess [input-ids attention-mask out-token-num] [cpu-mem-info])
                   answer! (runner* sess)]
      (memcpy-host! (long-pointer [5]) out-token-num-data)
      (memcpy-host! (float-pointer [1 1 1]) attention-mask-data)
      (memcpy-host! (long-pointer [8642, 562, 318]) input-ids-data)
      ;(synchronize! hstream)
      (answer! data-binding)
      ;(synchronize! hstream)
      (println 
       (pointer-vec (capacity! (long-pointer (mutable-data (first (bound-values data-binding)))) 14))))))

      ;=> [8642 562 318 407 262 691 835 284 8642 562 318 407 262 691]
      ;
;; Grass is not the only way to Grass is not the only way to