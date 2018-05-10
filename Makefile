all: dockerbuild

dockerbuild:
	sudo docker build -t thesis -f Dockerfile .

dockertest: dockerbuild
	sudo docker run --rm thesis python -m pytest -v -s tests/

pytest:
	pipenv run python -m pytest -v -s

install:
	pipenv install
