FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /app/
COPY minikube/app.py /app/app.py
COPY minikube/templates /app/templates
COPY minikube/static /app/static

RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python","app.py" ]