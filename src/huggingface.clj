(ns huggingface
  (:require
   [clj-http.client :as http]
   [clojure.java.io :refer [output-stream]]
   [clojure.string :as str]
   [clojure.java.io :as io]
   [progress.determinate]
   [jansi-clj.core]
   )
  (:import
   (org.apache.commons.io.input CountingInputStream)
   ))


(def spinner-style 
  ;(:ascii-basic progress.determinate/styles)
  (:coloured-ascii-boxes progress.determinate/styles)
  )

(defn- insert-at
  "Addes value into a vector at an specific index."
  [v idx val]

  (-> (subvec v 0 idx)
      (conj val)
      (into (subvec v idx))))


(defn- wrap-downloaded-bytes-counter
  "Middleware that provides an CountingInputStream wrapping the stream output"
  [client]
  (fn [req]
    (let [resp (client req)

          counter (CountingInputStream. (:body resp))]
      (merge resp {:body                     counter
                   :downloaded-bytes-counter counter}))))



(defn- download-with-progress [url target path authorization-token]
  (http/with-middleware
    (-> http/default-middleware
        ;(insert-after http/wrap-redirects wrap-downloaded-bytes-counter)
        (conj wrap-downloaded-bytes-counter)
        (conj http/wrap-lower-case-headers))
    (let [response (http/get url
                             {:as :stream
                              :headers {"Authorization"
                                        (format "Bearer %s" authorization-token)}})

          length (Long. (get-in response [:headers "content-length"] 0))
          buffer-size (* 1024 1024)
          progress (atom 0)]

      
      (progress.determinate/animate!
       progress
       :opts {:total (int (/ length 1024 1024))
              :redraw-rate 60 ; Use 60 fps for the demo
              :style spinner-style
              :label path
              :preserve true
              :units "MB"
              :line 3
              }
       (with-open [input (:body response)
                   output (output-stream target)]
         (let [buffer (make-array Byte/TYPE buffer-size)
               counter (:downloaded-bytes-counter response)]

           (loop []
             (let [size (.read input buffer)]
               (when (pos? size)
                 (.write output buffer 0 size)
                 (reset! progress (int (/ (.getByteCount counter) 1024 1024)))
                 (recur)))))))
      )))


(defn download-onnx-model-dir!
  "Download all files from a given 'model' from huggigface to a local dir"

  [models-base-dir model-name authorization-token]
  (let [split-by-slash (str/split model-name #"/")
        model-namespace (first split-by-slash)
        model-repo (second split-by-slash)
        model-base-dir (format "%s/%s/%s" models-base-dir model-namespace model-repo)

        model-files
        (->
         (http/get (format "https://huggingface.co/api/models/%s/%s/tree/main?recursive=true" model-namespace model-repo)
                   {:headers {"Authorization" (format "Bearer %s" authorization-token)}
                    :as :json})
         :body)
        progress (atom 0)]
    (progress.determinate/animate!
     progress 
     :opts {:total (count model-files)
            :line 2
            :style spinner-style}
     (run!
      (fn [{:keys [type path]}]
        (case type
          "directory" (io/make-parents (format "%s/%s" model-base-dir path))
          "file"
          (do
            (io/make-parents (format "%s/%s" model-base-dir path))
            (download-with-progress (format "https://huggingface.co/%s/%s/resolve/main/%s" model-namespace model-repo path)
                                    (format "%s/%s" model-base-dir path)
                                    path
                                    authorization-token)
            (swap! progress inc)
            )
          ))
      model-files))))



(comment
  
  
  (jansi-clj.core/erase-screen!)
  
  
  
  
  
  (download-onnx-model-dir!  "/hf-models"
                             "nvidia/Gemma-2b-it-ONNX-INT4"
                             (slurp "auth.txt"))
  )
  
    

