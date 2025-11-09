(ns huggingface
  (:require
   [clj-http.client :as http]
   [clojure.java.io :refer [output-stream]]
   [clojure.string :as str])
  (:import
   (org.apache.commons.io.input CountingInputStream)))



(defn print-progress-bar
  "Render a simple progress bar given the progress and total. If the total is zero
   the progress will run as indeterminated."
  ([progress total] (print-progress-bar progress total {}))
  ([progress total {:keys [bar-width]
                    :or   {bar-width 50}}]
   (if (pos? total)
     (let [pct (/ progress total)
           render-bar (fn []
                        (let [bars (Math/floor (* pct bar-width))
                              pad (- bar-width bars)]
                          (str (str/join (repeat bars "="))
                               (str/join (repeat pad " ")))))]
       (print (str "[" (render-bar) "] "
                   (int (* pct 100)) "% "
                   progress "/" total)))
     (let [render-bar (fn [] (str/join (repeat bar-width "-")))]
       (print (str "[" (render-bar) "] "
                   progress "/?"))))))

(defn insert-at
  "Addes value into a vector at an specific index."
  [v idx val]

  (-> (subvec v 0 idx)
      (conj val)
      (into (subvec v idx))))

(defn insert-after
  "Finds an item into a vector and adds val just after it.
   If needle is not found, the input vector will be returned."
  [v needle val]
  (let [index (.indexOf v needle)]
    (if (neg? index)
      v
      (insert-at v (inc index) val))))

(defn wrap-downloaded-bytes-counter
  "Middleware that provides an CountingInputStream wrapping the stream output"
  [client]
  (fn [req]
    (let [resp (client req)

          counter (CountingInputStream. (:body resp))]
      (merge resp {:body                     counter
                   :downloaded-bytes-counter counter}))))

(defn download-with-progress [url target authorization-token]
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
          buffer-size (* 1024 1024)]
      
      (println)
      (with-open [input (:body response)
                  output (output-stream target)]
        (let [buffer (make-array Byte/TYPE buffer-size)
              counter (:downloaded-bytes-counter response)]
          
          (loop [num-calls 0]
            (let [size (.read input buffer)]
              (when (pos? size)
                (.write output buffer 0 size)

                (when (or (zero? num-calls)
                          (= 0 (mod num-calls 1000)))
                  

                  (print-progress-bar
                   (if (some? counter)
                     (Math/round (/ (.getByteCount counter) 1.0))
                     0)
                   (Math/round (/ length 1.0)))
                  (println)
                  )
                (recur (inc num-calls)))))))
      (println))))


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
         :body)]
    (run!
     (fn [{:keys [type path]}]
       (case type
         "directory" (clojure.java.io/make-parents (format "%s/%s" model-base-dir path))
         "file"
         (do
           (clojure.java.io/make-parents (format "%s/%s" model-base-dir path))
           (download-with-progress (format "https://huggingface.co/%s/%s/resolve/main/%s" model-namespace model-repo path)
                                   (format "%s/%s" model-base-dir path)
                                   authorization-token))))
     model-files)))

(comment 
  (download-onnx-model-dir!  "/tmp/models" 
                             "onnx-community/text_summarization-ONNX" 
                             "<token>")
  )  

