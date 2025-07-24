# Use an official Python image as the base
FROM python:3.13.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "conversational_assitent.py"]
