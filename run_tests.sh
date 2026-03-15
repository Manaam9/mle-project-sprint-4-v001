#!/bin/bash

echo "Running service tests..."

source env_recsys_start/bin/activate

python test_service.py | tee test_service.log

echo "Tests finished. Log saved to test_service.log"
