# Thesis

[![Build Status](https://travis-ci.com/giuscri/thesis.svg?token=bzmoaCvPF1vTKtRyezHu&branch=master)](https://travis-ci.com/giuscri/thesis)
[![Coverage Status](https://coveralls.io/repos/github/giuscri/thesis/badge.svg)](https://coveralls.io/github/giuscri/thesis)

## Docker
`make dockerbuild` builds a docker image tagged as `thesis` with the needed
dependencies installed at system level. You can run bash inside a container and
run scripts there. For example:
```
sudo systemctl start docker # if you're using systemd
make dockerbuild
sudo docker run --rm -it thesis /bin/bash
```
If you still want to run scripts outside containers, see below.

## Install dependencies
```
pip install pipenv==11.10.1
make install
```

## How to test
```
make pytest
```

## How to run
```
pipenv run ./princeton.py --help
```

