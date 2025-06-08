# The python version in the sandbox should be the same as in the host so that cloudpickle works reliably
ARG PYTHON_VERSION=3.11.6
# The path to the workspace directory on the host file system
ARG WORKSPACE_ROOT
# The setup file will get run from without the workspace directory
ARG SETUP_FILE

FROM docker.io/python:${PYTHON_VERSION}
COPY ${SETUP_FILE} /tmp/setup.sh
COPY ${WORKSPACE_ROOT} /workspace

WORKDIR /workspace

RUN if [ -z "$SETUP_FILE" ]; then \
        echo "No setup file provided."; \
    else \
        echo "Using setup file: $SETUP_FILE" && \
        /bin/bash /tmp/setup.sh; \
    fi