FROM python:3.8

EXPOSE 8888

WORKDIR /app


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src src
COPY main.py .
COPY recourses .

CMD ["python", "main.py"]
