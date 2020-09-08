FROM julia:1.5

WORKDIR /workdir

COPY src ./src
COPY scripts ./scripts
COPY avi ./avi
COPY dev ./dev

COPY Project.toml ./
COPY Manifest.toml ./

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

RUN pwd
RUN ls 
RUN ls avi
RUN julia -e 'include("avi/bnn.jl")'

