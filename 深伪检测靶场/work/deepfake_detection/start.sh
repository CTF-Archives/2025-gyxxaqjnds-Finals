#!/bin/bash

if [ ! -d "/venv" ]; then
    echo "unpacking venv..."
    tar -xzvf /venv.tar.gz -C /
    rm /venv.tar.gz
fi

echo "Starting application..."

# If CMD is provided in the Dockerfile or during the container start, it will be passed as arguments to this script

if [ "$#" -eq 0 ]; then
    exec tail -f /dev/null
else
    exec "$@"
fi

