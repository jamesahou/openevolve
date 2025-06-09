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

COPY ${PROJECT_ROOT} /home/workspace
COPY ${SETUP_RELPATH} /home/setup.sh
COPY ${EVAL_RELPATH} /home/eval.py

# Install git
RUN apt-get update && \
    apt-get install -y git

    # Upgrade pip to the latest version
RUN pip install --upgrade pip

# Clone the main entry point gist
RUN git clone https://gist.github.com/93919a6ff1fe629b94ebc8c67a747059.git /home/openevolve_gists
RUN mv /home/openevolve_gists/openevolve_sandbox_main.py /home/main.py
RUN rm -rf /home/openevolve_gists

# Clone the OpenEvolve repository
RUN git clone https://github.com/ryanrudes/openevolve.git /home/openevolve

# Install OpenEvolve
RUN cd /home/openevolve && \
    pip install -e .

RUN mkdir /home/inputs
RUN mkdir /home/outputs
RUN mkdir /home/logs

WORKDIR /home/workspace

RUN if [ -z "$SETUP_FILE" ]; then \
        echo "No setup file provided."; \
    else \
        echo "Using setup file: $SETUP_FILE" && \
        cd /home/workspace && \
        /bin/bash /home/setup.sh; \
    fi

WORKDIR /