# Security

The program reads public model metadata from Hugging Face and, with `--this-pc`, runs `nvidia-smi`; it starts no
server and sends no data anywhere else. A Hugging Face token (`HF_TOKEN` or the one saved by `hf auth login`) is used
only for gated models, and is never printed or stored.

To report a vulnerability, please use GitHub's private vulnerability reporting
([report a vulnerability](https://github.com/nemmusu/llm-capacity-planner/security/advisories/new), or the **Security**
tab of the repository) rather than a public issue. Fixes go into the next release; only the latest release
is supported.
