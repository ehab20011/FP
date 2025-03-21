#Use a lightweight Python image
FROM python:3.11-slim

#Set the working directory in the container
WORKDIR /app

#Copy the requirements file
COPY requirements.txt /app/requirements.txt

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Create assets directory with proper permissions
RUN mkdir -p /app/assets && chmod 777 /app/assets

#Copy the rest of the application
COPY . /app/

#Expose the port Flask runs on (default: 5000)
EXPOSE 5000

#Set the default command to run your Flask app
CMD ["python", "flask_app.py"]