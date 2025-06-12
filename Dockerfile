# Gunakan official Python slim image
FROM python:3.11-slim

# Update dan install libgl1 (dependency OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0


WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file project
COPY . .

# Pastikan port dari environment variable di Railway
ENV PORT=8000

# Expose port sesuai PORT environment
EXPOSE ${PORT}

# Jalankan aplikasi Flask dengan python
CMD ["python", "app.py"]
