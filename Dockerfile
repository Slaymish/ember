# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /ember

# Install dependencies (using conda instead of micromamba)
COPY requirements_conda.txt /ember/
RUN conda install -y --channel conda-forge --file requirements_conda.txt && \
    conda clean --all --yes

# Copy all files
COPY . /ember

# Handle permissions explicitly
RUN chmod -R 755 /ember && chown -R 1000:1000 /ember

# Switch to non-root user (optional: make this configurable)
USER 1000:1000

# Set environment variables
ENV PYTHONPATH=/ember

# Default command (adjust if needed)
CMD ["bash"]
