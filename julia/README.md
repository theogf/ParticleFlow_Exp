# Running the code on EC2

Start an ec2 ubuntu instance with an associated ssh key 
(you will get prompted for one, create one if you don't have one yet)

If you download a new .pem file in the above steps reduce the permissions with

```bash
chmod 600 $PATH_TO_FILE.pem
```

This will avoid an error when running the ssh command below



```bash
# for example ssh -i .ssh/ec2.pem ubuntu@ec2-52-58-52-178.eu-central-1.compute.amazonaws.com
ssh -i $PATH_TO_FILE.pem ubuntu@$EC2URL
git clone https://github.com/theogf/ParticleFlow_Exp.git
# login with github credentials when prompted
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.1-linux-x86_64.tar.gz
tar zxvf julia-1.5.1-linux-x86_64.tar.gz
sudo ln -s $(pwd)/julia-1.5.1/bin/julia /bin/julia
cd ParticleFlow_Exp/julia
git clone https://github.com/theogf/AdvancedVI.jl --branch gaussparticleflow --single-branch dev/AdvancedVI
# login with github credentials if prompted
julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.develop(path="./dev/AdvancedVI"); Pkg.instantiate()'
```

Now run a script, for example

```bash
julia avi/run_gaussians.jl 
```

And any files saved can be copied out with scp as usual.

