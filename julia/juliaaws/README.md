This folder contains the code for running Julia experiments on Aws Batch

## Set up AWS account

Make sure you have a working AWS account, ideally with admin permissions.
Check that a simple command such as `aws s3 ls` gives the expected result.

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

