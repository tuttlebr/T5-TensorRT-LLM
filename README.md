# TensorRT-LLM Encoder/Decoder on Triton Inference Server

## Getting Started

**Fetch the Sources**

```bash
git submodule update --init --recursive
git lfs install
git lfs pull
```

**Build the Images**

`docker compose build trt-llm-backend`

`docker compose build triton-backend`

`docker compose build triton-trt-llm triton-client`

**Download Model**

`docker compose up download`

**Build TensorRT-LLM Engine**

`docker compose up build`

**Run Client and Server**

Update the URL of where you're hosting your Triton Server (`hostname -I`) in `.env`.
`docker compose up triton-server triton-client`
