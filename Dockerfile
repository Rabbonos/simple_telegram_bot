# Use the Microsoft Python image for Windows
FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Remove the original files in the whisper directory
RUN rm /usr/local/lib/python3.12/site-packages/whisper/audio.py
RUN rm /usr/local/lib/python3.12/site-packages/whisper/transcribe.py

# Copy new files to the whisper directory
COPY audio.py /usr/local/lib/python3.12/site-packages/whisper/
COPY transcribe.py /usr/local/lib/python3.12/site-packages/whisper/

# Remove the copied files from the working directory
RUN rm /app/audio.py
RUN rm /app/transcribe.py



# Ensure the app binds to port 8080
ENV PORT 8080

# Run the application
CMD ["python", "robotus.py"]
