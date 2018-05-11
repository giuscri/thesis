all: dockerbuild

dockerbuild:
	sudo docker build -t thesis -f Dockerfile .

dockertest: dockerbuild
	sudo docker run --rm thesis python -m pytest -sv --cov=. --cov-config=.coveragerc tests/

pytest:
	pipenv run python -m pytest -v -s

install:
	pipenv install

clean:
	for f in `cat .dockerignore`; do rm -rf $$f; done
