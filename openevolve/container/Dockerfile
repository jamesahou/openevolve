# /
# ├── setup.sh          # The setup script that installs dependencies and sets up the environment
# ├── eval.py           # The evaluation script that will be run in the container
# ├── main.py           # The main entry point for the evaluation
# ├── workspace/        # The workspace directory containing the project code
# ├── imps/             # Directory for implementation files (mounted as read-only)
# ├── inputs/           # Directory containing the test case files
#     ├── 1.pickle
#     ├── 2.pickle
#     └── ...
# ├── outputs/          # Directory for output files
#     ├── implementation_id/
#         ├── test_1.pickle
#         ├── test_2.pickle
#         └── ...
# └── logs/             # Directory for logs
#     └── implementation_id/
#         ├── test_1/
#             ├── stdout.txt
#             └── stderr.txt
#         └── ...
#     └── ...

# The python version in the sandbox should be the same as in the host so that cloudpickle works reliably
ARG PYTHON_VERSION=3.11.6

# Use the official Python image as the base image
FROM docker.io/python:${PYTHON_VERSION}

# The absolute path to the project root directory on the host file system
ARG PROJECT_ROOT
# The setup file will get run from without the workspace directory
ARG SETUP_RELPATH
# The evaluation entry point script
ARG EVAL_RELPATH

COPY ${PROJECT_ROOT} /workspace
COPY ${SETUP_RELPATH} /setup.sh
COPY ${EVAL_RELPATH} /eval.py

# Install git
RUN apt-get update && \
    apt-get install -y git

# Clone the main entry point gist
RUN git clone https://gist.github.com/93919a6ff1fe629b94ebc8c67a747059.git /openevolve_gists
RUN mv /openevolve_gists/openevolve_sandbox_main.py /main.py
RUN rm -rf /openevolve_gists

COPY ./funsearch/container/main.py /main.py

COPY . /openevolve
RUN cd /openevolve && \
    pip install -e .

RUN mkdir /inputs
RUN mkdir /outputs
RUN mkdir /logs

WORKDIR /workspace

RUN if [ -z "$SETUP_FILE" ]; then \
        echo "No setup file provided."; \
    else \
        echo "Using setup file: $SETUP_FILE" && \
        /bin/bash /setup.sh; \
    fi

WORKDIR /