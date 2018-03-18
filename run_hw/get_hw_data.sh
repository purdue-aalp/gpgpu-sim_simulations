export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
TITANX_PASCAL_DATA=$DATA_ROOT/TITAN-X-PASCAL
P100_PASCAL_DATA=$DATA_ROOT/TESLA-P100

if [ ! -d $TITANX_PASCAL_DATA ]; then
    echo "in"
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.titanx.cycle.tgz .
    tar -xzvf pascal.titanx.cycle.tgz -C $DATA_ROOT
    rm pascal.titanx.cycle.tgz
fi

if [ ! -d $P100_PASCAL_DATA ]; then
    echo "in"
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data/pascal.tesla.p100.cycles.tgz .
    tar -xzvf pascal.tesla.p100.cycles.tgz -C $DATA_ROOT
    rm pascal.tesla.p100.cycles.tgz
fi
