This folder contains the code for running Julia experiments on Aws Batch

## Set up AWS account

Make sure you have a working AWS account, ideally with admin permissions.
Check that a simple command such as `aws s3 ls` gives the expected result.

# Use notebooks

## Set up 

Start with setting up credentials with git

https://docs.aws.amazon.com/codecommit/latest/userguide/setting-up-ssh-unixes.html


Create repository $REPONAME in CodeCommit, and add the repository as a remote

```bash
$ git remote add aws ssh://git-codecommit.$REGION.amazonaws.com/v1/repos/$REPONAME
```

Push the current branch by running

```
$ git push aws
```

Follow the instructions from

https://d1.awsstatic.com/whitepapers/julia-on-sagemaker.pdf?did=wp_card&trk=wp_card

making sure to add the above CodeCommit repository


## TODO
* port dependencies to julia 1.1.1
* convert plotting to a supported library


# Alternative using AWS Batch (incomplete)

## Create the stack

To create the required AWS resources using the provided CloudFormation
template run the following script:

```bash
$ sh path/to/createawsresources.sh
```

Note: if you previously created and deleted a stack, the S3 bucket will have to be 
manually deleted before the same stack name
can be used. So either manually delete the S3 bucket or use a different name
by changing the file `stackname`.


## Build docker image

Install docker and make sure you have permissions to use docker commands without sudo,
on Mac OS X this should be the default on Ubuntu to do so you need to run 

```bash
$ sudo chmod 666 /var/run/docker.sock
```

Build the docker image using

```bash
$ path/to/rebuilddocker.sh
```

## TODO

* Install depenencies in dockerfile
* Create better endpoint with arguments in dockerfile
* Instructions for starting a batch job with CLI (or add script to start jobs)
