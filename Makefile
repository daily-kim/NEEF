default: build

help:
	@echo 'Management commands for NEEF:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t NEEF 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus all --ipc=host --name NEEF -v `pwd`:/workspace NEEF:latest /bin/bash

up: build run

rm: 
	@docker rm NEEF

stop:
	@docker stop NEEF

reset: stop rm
