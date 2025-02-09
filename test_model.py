import tensorflow as tf

model_path = "graph_opt.pb"

try:
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    print("✅ Model is a valid TensorFlow GraphDef.")
except Exception as e:
    print(f"❌ Error: {e}")
