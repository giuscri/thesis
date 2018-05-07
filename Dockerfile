FROM python:3.6-stretch

COPY . /app
WORKDIR /app

RUN pip install pipenv==11.10.1
RUN pipenv install

CMD ["pipenv", "run", "python", "-m", "pytest"]