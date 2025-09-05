# Base image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy folders
COPY api/ ./api/
COPY model/ ./model/
COPY collectors/ ./collectors/

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install -r api/requirements.txt

RUN pip install -r collectors/requirements.txt

RUN pip install -r model/requirements.txt

RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
