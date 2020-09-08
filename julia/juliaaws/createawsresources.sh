#!/usr/bin/env bash

CLOUDFORMATIONTEMPLATE=CloudFormation.template
STACKNAMEFILE=stackname
BASEDIR=$(dirname $0)
STACKNAME=$(cat $BASEDIR/$STACKNAMEFILE)

set -x
echo aws cloudformation create-stack --stack-name $STACKNAME --template-body file://$BASEDIR/$CLOUDFORMATIONTEMPLATE  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

