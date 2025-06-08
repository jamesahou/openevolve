"""
/
├── setup.sh          # The setup script that installs dependencies and sets up the environment
├── eval.py           # The evaluation script that will be run in the container
├── main.py           # The main entry point for the evaluation
├── workspace/        # The workspace directory containing the project code
├── implementations/  # Directory for implementation files
├── inputs/           # Directory containing the test case files
    ├── 1.pickle
    ├── 2.pickle
    └── ...
├── outputs/          # Directory for output files
    ├── implementation_id/
        ├── 1.pickle
        ├── 2.pickle
        └── ...
└── logs/             # Directory for logs
    └── implementation_id/
        ├── test_1/
            ├── stdout.txt
            └── stderr.txt
        └── ...
    └── ...
"""
# The python version in the sandbox should be the same as in the host so that cloudpickle works reliably
ARG PYTHON_VERSION=3.11.6
# The path to the workspace directory on the host file system
ARG WORKSPACE_ROOT
# The setup file will get run from without the workspace directory
ARG SETUP_FILE
# The evaluation entry point script
ARG EVAL_ENTRY_POINT

FROM docker.io/python:${PYTHON_VERSION}
COPY ${SETUP_FILE} /setup.sh
COPY ${WORKSPACE_ROOT} /workspace
COPY ${EVAL_FILE} /eval.py

COPY /Users/jameshou/Documents/Code/openevolve/ /openevolve
RUN cd /openevolve && \
    pip install -e .

RUN mkdir /implementations
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