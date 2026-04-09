FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

RUN dnf update -y && dnf install -y postgresql-devel gcc python3-devel make && dnf clean all

COPY requirements.lambda.txt /var/task/requirements.txt
RUN pip install --no-cache-dir -r /var/task/requirements.txt

COPY lambda_function.py /var/task/
COPY pipeline/ /var/task/pipeline/
COPY lambda_function_equity.py /var/task/

CMD ["lambda_function.handler"]