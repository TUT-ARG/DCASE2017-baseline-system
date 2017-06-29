help:
	@cat Makefile

DOCKER_NAME=dcase-env
DOCKER_FILE=Dockerfile
BACKEND=theano
SRC=$(shell dirname `pwd`)

build:
	docker build -t $(DOCKER_NAME) -f $(DOCKER_FILE) .

bash: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) bash

task1: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task1.py -s dcase2017,gmm -o; python applications/task1.py -show_eval"

task2: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task2.py -s dcase2017,gmm -o; python applications/task2.py -show_eval"

task3: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task3.py -s dcase2017,gmm -o; python applications/task3.py -show_eval"

task1_challenge: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task1.py -s dcase2017 -m challenge -o"

task2_challenge: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task2.py -s dcase2017 -m challenge -o"

task3_challenge: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task3.py -s dcase2017 -m challenge -o"

test1: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task1.py -s minimal; python applications/task1.py -show_eval"

test2: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task2.py -s minimal; python applications/task2.py -show_eval"

test3: build
	docker run -it -v $(SRC):/DCASE2017-baseline-system $(DOCKER_NAME) /bin/bash -c "python applications/task3.py -s minimal; python applications/task3.py -show_eval"
