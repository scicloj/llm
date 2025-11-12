(ns inference-gemma
  (:require
   [tech.v3.datatype.argops]
   [uncomplicate.clojure-cpp
    :refer [get-entry long-pointer pointer-pointer pointer-pointer
            short-pointer]]
   [uncomplicate.commons.core :as uc-co-core :refer [info with-release]]
   [uncomplicate.diamond.internal.onnxrt.constants]
   [uncomplicate.diamond.internal.onnxrt.core :refer [bound-values mutable-data value] :as onnxrt-internal]
   [uncomplicate.diamond.native])
  (:import
   [ai.djl.huggingface.tokenizers HuggingFaceTokenizer]
   [ai.onnxruntime.platform Fp16Conversions]))




(defn prompt->input-ids [^HuggingFaceTokenizer tokenizer ^String prompt]
  (seq
   (.. tokenizer
       (encode prompt)
       getIds)))

(defn generate-new-token
  "Take a list of tokens, and generaes the next one"
  [input-ids sess mem-info]

  (with-release
    [float-arr
     (float-array 256000)

     empty-tz
     (onnxrt-internal/onnx-tensor
      mem-info
      [1 1 0 256]
      (short-pointer [])
      :float16)

     input-ids-tz
     (onnxrt-internal/onnx-tensor
      mem-info
      [1 (count input-ids)]
      (long-pointer input-ids)
      :long)

     positions-ids-tz
     (onnxrt-internal/onnx-tensor
      mem-info
      [1 (count input-ids)]
      (long-pointer (range (count input-ids)))
      :long)
     attention-mask-tz
     (onnxrt-internal/onnx-tensor
      mem-info

      [1 (count input-ids)]
      (long-pointer (repeat (count input-ids) 1))
      :long)
     infer! (onnxrt-internal/runner* sess)
     my-binding (onnxrt-internal/io-binding sess)

     _ (do
         (onnxrt-internal/bind-input! my-binding "input_ids" input-ids-tz)
         (onnxrt-internal/bind-input! my-binding "position_ids" positions-ids-tz)
         (onnxrt-internal/bind-input! my-binding "attention_mask" attention-mask-tz)
         (onnxrt-internal/bind-output! my-binding "logits" mem-info)
         (run!
          (fn [idx]
            (onnxrt-internal/bind-input! my-binding (format "past_key_values.%s.key" idx) empty-tz)
            (onnxrt-internal/bind-input! my-binding (format "past_key_values.%s.value" idx) empty-tz))
          (range 18)))

     outputs
     (infer! my-binding)


     value-info
     (->
      (bound-values outputs)
      first
      value
      info)

     _ (println :logits-info value-info)


     logits
     (->
      (bound-values outputs)
      first
      value
      mutable-data
      short-pointer)

     logits-short-array
     (short-array
      (map
       #(short (get-entry logits %))
       (range (*  (count input-ids) 256000))))

     logits-short-buffer (java.nio.ShortBuffer/wrap logits-short-array)
     logits-float-buffer
     (Fp16Conversions/convertFp16BufferToFloatBuffer logits-short-buffer)]

    _ (.rewind logits-float-buffer)

    ;;todo, we only need the last
    (loop [tokens []]
      (if (not (.hasRemaining logits-float-buffer))
        tokens
        (let [_ (.get logits-float-buffer float-arr)
              token-idx (tech.v3.datatype.argops/argmax float-arr)]

          (recur (conj tokens token-idx)))))))

(defn generate-new-prompt [prompt tokenizer sess mem-info]
  (let [new-tokens (generate-new-token (prompt->input-ids tokenizer prompt) sess mem-info)
        new-prompt (str prompt (.decode tokenizer (long-array [(last new-tokens)])))]
    new-prompt))


(defn stream-completion [start-prompt max-tokens tokenizer ort-session mem-info new-prompt-call-back-fn]
  (loop [counter max-tokens
         prompt start-prompt]
    (when (pos? counter)
      (let [new-prompt (generate-new-prompt prompt tokenizer ort-session mem-info)]
        (new-prompt-call-back-fn new-prompt)
        (recur  (dec counter)

                new-prompt)))))

;; The following assumes tis model being accessible on local disk
;; it can be downloaded via
;;(huggingface/download-onnx-model-dir!  "/hf-models"
;;                             "nvidia/Gemma-2b-it-ONNX-INT4"
;;                             <hugging-face-token>)

(defn complete [{:keys [prompt max-tokens]}]
  (with-release [models-base-dir "/hf-models"
                 model-folder "nvidia/Gemma-2b-it-ONNX-INT4"
                 model-dir (format "%s/%s" models-base-dir model-folder)
                 tokenizer (-> "file:%s"
                               (format model-dir)
                               java.net.URI.
                               java.nio.file.Path/of
                               HuggingFaceTokenizer/newInstance)
                 model-file (format "%s/model.onnx" model-dir)
                 env (onnxrt-internal/environment nil)
                 opt (onnxrt-internal/options)
                 mem-info (onnxrt-internal/memory-info :cpu :arena 0 :default)
                 sess (onnxrt-internal/session env model-file opt)]
    (stream-completion prompt max-tokens tokenizer sess mem-info println)))



