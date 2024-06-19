FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code
COPY ./fn.py /code
COPY ./data /code/data

CMD ["python","app.py"]

# docker build -t dashtest .
# docker run -h localhost -p 8050:8050 -d --name [container_name] [image_name] 
# docker run --rm -p 8051:8050 --network host --name dashtest dashtest
# docker run --rm -p 8051:8050 --bind 0.0.0.0 --name dashtest dashtest    --bind 0.0.0.0     