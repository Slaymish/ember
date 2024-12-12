# Base image
FROM mambaorg/micromamba:0.25.1

# Set working directory
WORKDIR /ember

# Install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements_conda.txt /ember/
RUN micromamba install -y -n base --channel conda-forge --file requirements_conda.txt && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install torch

# Copy all files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /ember

# Switch to the non-root user
USER $MAMBA_USER

ENV PYTHONPATH=/ember