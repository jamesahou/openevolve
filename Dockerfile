FROM docker.io/python:3.11.6

WORKDIR /workspace

# Use PDM to keep track of exact versions of dependencies
RUN pip install pdm
COPY pyproject.toml README.md pdm.lock ./
# install dependencies first. PDM also creates a /workspace/.venv here.
ENV PATH="/workspace/.venv/bin:$PATH"
RUN pdm install  --no-self
COPY examples ./examples
COPY funsearch ./funsearch

RUN pip install --no-deps . && rm -r ./funsearch ./build

RUN pip install llm-gpt4all

COPY extra-openai-models.yaml /root/.config/io.datasette.llm/extra-openai-models.yaml

CMD /bin/bash