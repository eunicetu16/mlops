FROM python:3.9-slim  
WORKDIR /app

# Copy just the requirements first for caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app directory
COPY app/ .

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]

