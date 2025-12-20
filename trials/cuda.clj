;(System/setProperty "org.bytedeco.javacpp.logger.debug" "true")
;; (ns cuda
;;   (:require [uncomplicate.clojurecl.core :refer :all])
;;   (:require [uncomplicate.commons.core :refer [info]]))

(ns
 cuda
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [transfer! asum native]]
            [uncomplicate.diamond.tensor :refer [tensor with-diamond]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))


(with-release [cudnn (cudnn-factory)
               t (tensor cudnn [2 3 4 1] :float :nchw)]
  (transfer! (range) t)
  (println :sum-1 (asum t))
  (transfer! (map inc (range)) t)
  (println :sum-2 (asum t))
  (native (asum t))
  )

(with-diamond cudnn-factory []
  (with-release [t (tensor [2 3] :float :nc)]
    (transfer! (range) t)
    (asum t)))
