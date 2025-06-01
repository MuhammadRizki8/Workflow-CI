
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first for better caching
COPY MLProject/conda.yaml /app/conda.yaml

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ml_model_env", "/bin/bash", "-c"]

# Install additional packages
RUN pip install mlflow==2.8.1 dagshub==0.3.7

# Copy application code
COPY MLProject/ /app/

# Create necessary directories
RUN mkdir -p /app/preprocessed-csvs /app/models /app/reports /app/mlruns

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

# Expose MLflow UI port
EXPOSE 5000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
conda activate ml_model_env\n\
\n\
# Function to run training\n\
run_training() {\n\
    echo "Starting ML model training..."\n\
    python modelling.py "$@"\n\
    echo "Training completed!"\n\
}\n\
\n\
# Function to start MLflow UI\n\
start_mlflow_ui() {\n\
    echo "Starting MLflow UI..."\n\
    mlflow ui --host 0.0.0.0 --port 5000\n\
}\n\
\n\
# Check command line arguments\n\
if [ "$1" = "train" ]; then\n\
    shift\n\
    run_training "$@"\n\
elif [ "$1" = "ui" ]; then\n\
    start_mlflow_ui\n\
elif [ "$1" = "both" ]; then\n\
    shift\n\
    run_training "$@"\n\
    echo "Training completed. Starting MLflow UI..."\n\
    start_mlflow_ui\n\
else\n\
    echo "Usage:"\n\
    echo "  docker run <image> train [options]  - Run training only"\n\
    echo "  docker run <image> ui               - Start MLflow UI only"\n\
    echo "  docker run <image> both [options]   - Run training then start UI"\n\
    echo ""\n\
    echo "Training options:"\n\
    echo "  --tune          - Run with hyperparameter tuning"\n\
    echo "  --evaluate-only - Only evaluate existing model"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["train"]