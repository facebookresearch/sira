<!-- Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree. -->
# SIRA  

**S**uper**I**ntelligent **R**etrieval **A**gent

SIRA is a multi-stage retrieval pipeline that uses LLMs to enrich both documents and queries, improving BM25 retrieval quality without training. The pipeline consists of five stages: data preparation, BM25 indexing, corpus enrichment (LLM-generated indexing phrases for documents), query expansion (LLM-generated search terms), and LLM-based pointwise reranking. SIRA delivers large gains over BM25 on BEIR benchmarks using only inference-time compute.

Note on reported numbers: each dataset is run with its own prompt and pipeline configuration (see `scripts/configs/`), so the results are not a single-configuration benchmark and are not directly comparable to systems evaluated under one fixed setup.

Paper: https://arxiv.org/pdf/2605.06647
## Requirements

- Python >= 3.12
- CUDA-capable GPU(s) (tested on NVIDIA H100)
- Rust toolchain (for building the bm25x extension)
- Conda (recommended for environment management)

## Setup

```bash
# Create and activate the conda environment
conda create -n sira312 python=3.12 -y
conda activate sira312
pip install -e .

# Activate the development sandbox
source sandbox.sh
```

## Quick Start

```bash
# Run the full pipeline on a single dataset (auto-starts LLM server)
python scripts/run_pipeline.py data=scifact server.auto_start=true

# Run on multiple datasets
python scripts/run_pipeline.py datasets='[scifact,arguana,fiqa]' server.auto_start=true

# Run specific stages only
python scripts/run_pipeline.py data=scifact stages='[enrich_query,rerank]'
```

See [scripts/README.md](scripts/README.md) for the full pipeline documentation, configuration options, and data layout.

## Citation

```bibtex
@article{yang2026sira,
  title={Superintelligent Retrieval Agent: The Next Frontier of Information Retrieval},
  author={Yang, Zeyu and Ma, Qi and Chen, Jason and Shrivastava, Anshumali},
  journal={arXiv preprint arXiv:2605.06647},
  year={2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The `src/sira/bm25x/` directory contains code derived from [bm25x](https://github.com/lightonai/bm25x) by LightOn, licensed under Apache 2.0. See the [NOTICE](src/sira/bm25x/NOTICE) file for details.
