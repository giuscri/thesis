FROM python:3.6-stretch

COPY . /app
WORKDIR /app

RUN pip install --verbose pipenv==11.10.1
RUN pipenv install --verbose

CMD ["pipenv", "run", "python", "-m", "pytest"]