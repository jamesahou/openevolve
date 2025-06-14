<!--
<p align="center">
  <img src="https://raw.githubusercontent.com/ryanrudes/openevolve/main/docs/logo.png" alt="OpenEvolve Logo" width="200"/>
</p>
-->

<h1 align="center">OpenEvolve</h1>

<p align="center">
  <b>Codebase-scale open source evolutionary program synthesis inspired by <a href="https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/">AlphaEvolve</a></b>
</p>

---

## Overview

OpenEvolve is an open-source framework for evolutionary program synthesis, inspired by the <a href="https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/">AlphaEvolve</a> project by DeepMind. It leverages evolutionary algorithms and large language models to automatically generate, mutate, and evaluate code, enabling the discovery of novel solutions to programming tasks.

<!--
<p align="center">
  <img src="" alt="OpenEvolve Architecture" width="600"/>
</p>
-->

---

## Key Features

- **Codebase-Scale Optimization** Contrary to previous implementations, OpenEvolve allows for codebase-scale optimization, as opposed to just one function.
- **Evolutionary Search**: Uses evolutionary strategies to explore the program space.
- **LLM Integration**: Incorporates large language models for code mutation and synthesis.
- **Asynchronous Evaluation**: Runs and evaluates generated code asynchronously in a sandboxed environment.
- **Extensible**: Easily add new tasks and evaluation metrics.

---

## Installation

OpenEvolve requires Docker to run ([install instructions](https://docs.docker.com/engine/install/)). For the remaining requirements, please follow the steps:

```bash
# Clone the repository
git clone https://github.com/ryanrudes/openevolve.git
cd openevolve

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

---

## Citing OpenEvolve

If you use this project, please cite it as such:

```
@software{openevolve,
  title        = {OpenEvolve: Codebase-scale evolutionary algorithms for code optimization, powered by LLMs},
  author       = {Ryan Rudes, James Hou, and Aditya Mehta},
  year         = {2025},
  howpublished = {\url{https://github.com/ryanrudes/openevolve}},
  note         = {Open-source evolutionary program synthesis framework. Based on original code licensed under Apache License 2.0 by the FunSearch project.}
}
```

---

## License

This project is licensed under the terms of the MIT license. See [LICENSE.txt](LICENSE.txt) for details.

---

<p align="center">
  <i>OpenEvolve: Codebase-scale evolutionary algorithms for code optimization, powered by LLMs.</i>
</p>
