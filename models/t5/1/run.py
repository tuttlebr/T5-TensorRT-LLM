import json
from pathlib import Path

import tensorrt as trt
import torch


import tensorrt_llm
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    world_size = config["builder_config"]["tensor_parallel"]
    assert (
        world_size == tensorrt_llm.mpi_world_size()
    ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
    num_heads = config["builder_config"]["num_heads"] // world_size
    hidden_size = config["builder_config"]["hidden_size"] // world_size
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    cross_attention = config["builder_config"]["cross_attention"]
    has_position_embedding = config["builder_config"]["has_position_embedding"]
    has_token_type_embedding = config["builder_config"][
        "has_token_type_embedding"]
    num_kv_heads = num_heads

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
    )

    dtype = config["builder_config"]["precision"]
    max_input_len = config["builder_config"]["max_input_len"]

    return model_config, world_size, dtype, max_input_len


class TRTLLMEncDecModel:

    def __init__(self, engine_name, engine_dir, debug_mode=False):
        engine_dir = Path(engine_dir)

        # model config
        encoder_config_path = engine_dir / "encoder" / "config.json"
        encoder_model_config, world_size, dtype, max_input_len = read_config(
            encoder_config_path)
        decoder_config_path = engine_dir / "decoder" / "config.json"
        decoder_model_config, _, _, _ = read_config(decoder_config_path)
        self.encoder_model_config = encoder_model_config
        self.decoder_model_config = decoder_model_config

        # MGMN config
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        # load engine
        engine_name = get_engine_name(engine_name, dtype, world_size,
                                      runtime_rank)
        with open(engine_dir / "encoder" / engine_name, "rb") as f:
            encoder_engine_buffer = f.read()
        with open(engine_dir / "decoder" / engine_name, "rb") as f:
            decoder_engine_buffer = f.read()

        # session setup
        self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
            encoder_engine_buffer)
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)
        self.stream = torch.cuda.current_stream().cuda_stream

    @classmethod
    def from_engine(cls, engine_name, engine_dir, debug_mode=False):
        return cls(engine_name, engine_dir, debug_mode=debug_mode)

    def encoder_run(self,
                    input_ids,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False):
        batch_size = input_ids.shape[0]
        input_lengths = torch.tensor([len(x) for x in input_ids],
                                     dtype=torch.int32,
                                     device='cuda')
        max_input_length = torch.max(input_lengths).item()

        # set input tensors and shapes
        inputs = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
        }
        if self.encoder_model_config.has_position_embedding:
            inputs['position_ids'] = position_ids
        if self.encoder_model_config.has_token_type_embedding:
            inputs['token_type_ids'] = token_type_ids
        for k, v in inputs.items():
            self.encoder_session.context.set_input_shape(k, v.shape)

        # set output tensors and shapes
        outputs = {
            'encoder_output':
            torch.empty((batch_size, max_input_length,
                         self.encoder_model_config.hidden_size),
                        dtype=trt_dtype_to_torch(
                            self.encoder_session.engine.get_tensor_dtype(
                                'encoder_output')),
                        device='cuda')
        }

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(
                        name
                ) == trt.TensorIOMode.OUTPUT and name not in outputs.keys():
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device='cuda')
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        ok = self.encoder_session.run(inputs, outputs, self.stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

        # -------------------------------------------
        if debug_mode:
            torch.cuda.synchronize()
            # use print_tensor() to print the tensors registered in the encoder network
            print("--------------------------------------")
            print("Debug output for Encoder")
            print("--------------------------------------")
            print("Registered output tensors are: ", outputs.keys())
            print_tensor('encoder_output', outputs['encoder_output'])
            print("--------------------------------------")
        # -------------------------------------------

        return outputs

    def generate(
        self,
        encoder_input_ids,
        decoder_input_ids,
        max_new_tokens,
        num_beams=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        debug_mode=False,
    ):
        # encoder run
        encoder_input_lengths = torch.tensor(
            [len(x) for x in encoder_input_ids],
            dtype=torch.int32,
            device='cuda')
        encoder_outputs = self.encoder_run(encoder_input_ids,
                                           debug_mode=debug_mode)
        encoder_output = encoder_outputs['encoder_output']
        torch.cuda.synchronize()

        # decoder_batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [len(x) for x in decoder_input_ids],
            dtype=torch.int32,
            device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            encoder_max_input_length=encoder_output.shape[1])

        torch.cuda.synchronize()
        output_ids = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
        )
        torch.cuda.synchronize()

        return output_ids