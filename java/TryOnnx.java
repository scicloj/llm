import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class TryOnnx {


public static void main(String[] args) throws Exception {
    var env = OrtEnvironment.getEnvironment();
    var session = env.createSession("models/gpt-oss-20b/cuda/model.onnx",new OrtSession.SessionOptions());


	}
}