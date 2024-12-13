# Base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /ember

# Install mamba and dependencies
COPY requirements_conda.txt /ember/
RUN conda install -y -n base -c conda-forge mamba && \
    mamba install -y -n base python=3.8 py-lief pytorch torchvision torchaudio -c pytorch -c conda-forge && \
    mamba install -y -n base --file requirements_conda.txt && \
    conda clean --all --yes

# Add 'pynvml' for GPU monitoring
RUN pip install pynvml

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
