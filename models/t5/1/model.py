import json
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import logging
import triton_python_backend_utils as pb_utils
import numpy as np

from run import TRTLLMEncDecModel

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TritonPythonModel:
    def initialize(self, args):
        
        self.model_config = json.loads(args["model_config"])
        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "output_text"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            self.output0_config["data_type"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.hf_model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda().eval()
        self.tllm_model = TRTLLMEncDecModel.from_engine("enc_dec", "/trt_engines")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_text_batch = pb_utils.get_input_tensor_by_name(request, "input_text").as_numpy()
            max_new_tokens_batch = pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()
            num_beams_batch = pb_utils.get_input_tensor_by_name(request, "num_beams").as_numpy()

            for input_text, max_new_tokens, num_beams in zip(input_text_batch, max_new_tokens_batch, num_beams_batch):
                input_text = input_text[0].decode("utf-8")
                max_new_tokens = max_new_tokens[0]
                num_beams = num_beams[0]

                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.type(torch.IntTensor).cuda()
                decoder_input_ids = torch.IntTensor([[self.hf_model.config.decoder_start_token_id]]).cuda()

                tllm_output_ids = self.tllm_model.generate(
                    encoder_input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
                
                decoded = self.tokenizer.decode(tllm_output_ids[0][0], skip_special_tokens=True)
                logger.info(f"{input_text}\t{decoded}")                
                responses.append(decoded)

        decoded = pb_utils.Tensor("output_text", np.asarray(responses, dtype=object))

        inference_responses = pb_utils.InferenceResponse(
            output_tensors=[decoded]
        )
        return [inference_responses]

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger.info("Postprocess cleaning up...")