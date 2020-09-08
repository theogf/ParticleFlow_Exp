#!/usr/bin/env bash

STACKNAMEFILE=stackname
BASEDIR=$(dirname $0)
STACKNAME=$(cat $BASEDIR/$STACKNAMEFILE)
DOCKERFILE=$BASEDIR/../Dockerfile
REGION=$(aws configure get region)
ACCOUNTID=$(aws sts get-caller-identity --query 'Account' --output text)

set -e
set -x

echo $(whoami)

IMAGE_TAG=$STACKNAME
IMAGE_URL=$ACCOUNTID.dkr.ecr.$REGION.amazonaws.com/$STACKNAME:$STACKNAME

echo "Building the docker image"
if [ "$1" == "quick" ]; then
    DOCKERFILE=Dockerfile-quick
    ADDITIONAL_ARGS="--build-arg base_image=$IMAGE_URL"
else
    DOCKERFILE=Dockerfile
    ADDITIONAL_ARGS="--no-cache=true"
fi
docker build -t $IMAGE_TAG -f $(dirname $0)/$DOCKERFILE $ADDITIONAL_ARGS $(dirname $0)/..

exit 0

echo "Uploading the image to ECR"
$(aws ecr get-login --no-include-email)
docker tag $IMAGE_TAG $IMAGE_URL
docker push $IMAGE_URL

echo "Image successfully pushed"
