# Use the NVIDIA NeMo container as the base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

RUN ls

# Copy the current directory contents into the container at /app
COPY ../../.. /app

RUN ls

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry install

# Ensure that the script runs on GPU (uncomment if needed)
# ENV CUDA_VISIBLE_DEVICES=0

# Command to run when the container starts
CMD ["bash"]

