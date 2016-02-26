#!/usr/bin/env bash

docker pull bekozi/ntest-ubuntu
docker rm ocgis-test-runner
docker run -id --name ocgis-test-runner bekozi/ntest-ubuntu
docker exec ocgis-test-runner bash -c "conda create -y -n ocgis -c nesii/channel/ocgis -c nesii ocgis==1.2.1 nose"
docker exec ocgis-test-runner bash -c "source activate ocgis && python -c \"from ocgis.test import run_simple; run_simple()\""
docker stop ocgis-test-runner