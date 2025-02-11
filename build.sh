#!/bin/bash

echo -n "Tag name: "
read -r tag
docker build -t $tag .