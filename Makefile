all: dockerbuild

dockerbuild:
	git ls-files --directory --ignored --exclude-standard --others | tee .dockerignore
	sudo docker build -t thesis -f Dockerfile .

dockertest: dockerbuild
	sudo docker run --rm thesis python -m pytest -sv --cov=. --cov-config=.coveragerc tests/

pytest: lint
	pipenv run python -m pytest -v -s

install:
	pipenv install

clean:
	[[ -f .dockerignore ]] && rm -rf `cat .dockerignore`

lint:
	black .

protect:
	sudo chattr -R +i model/ mnist/ fast-gradient-sign/

unprotect:
	sudo chattr -R -i model/ mnist/ fast-gradient-sign/
