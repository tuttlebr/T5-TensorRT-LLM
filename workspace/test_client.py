import logging
import numpy as np
import os
import tritonclient.grpc as grpcclient


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "0.0.0.0")
URL = f"{TRITON_SERVER_URL}:8001"

if __name__ == "__main__":
    client = grpcclient.InferenceServerClient(url=URL, verbose=False)

    input_text = ["Translate English to German: What is a computer?"]
    max_new_tokens = [64]
    num_beams = [3]

    input_text = np.array(input_text, dtype='|S64').reshape((-1,1))
    max_new_tokens = np.array(max_new_tokens).reshape((-1,1))
    num_beams = np.array(num_beams).reshape((-1,1))

    inputs = [
        grpcclient.InferInput("input_text", input_text.shape, "BYTES"),
        grpcclient.InferInput("max_new_tokens", max_new_tokens.shape, "INT64"),
        grpcclient.InferInput("num_beams", num_beams.shape, "INT64")
    ]

    inputs[0].set_data_from_numpy(input_text)
    inputs[1].set_data_from_numpy(max_new_tokens)
    inputs[2].set_data_from_numpy(num_beams)

    triton_outputs = [grpcclient.InferRequestedOutput("output_text")]

    infer_result = client.infer(
                    "t5",
                    inputs,
                    model_version="1",
                )

    logger.info(str(infer_result.as_numpy("output_text").astype(str)))