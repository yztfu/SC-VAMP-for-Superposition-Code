# cd /home/tfu/Documents/AMP/GOAMP_SC_GLM

NEW_PATH="./output/"

if [ -d "$NEW_PATH" ]; then
        echo "PATH already exists."
else
        echo "PATH does not exist. Creating it..."
        mkdir -p "$NEW_PATH"
        echo "PATH has been created successfully."
fi

BASE_ARGS="
    --algo_ori
    --algo_sc
    --mat_type dct
    --N_se 65536
    --T 200
    --num_trials 10
    --save_interval 1
"

SIGNAL_ARGS="
    --x_prior_type src
    --B_src 16
"

CODE_ARGS="
    --N 1024
    --rate 1.6
"

NOISE_ARGS="
    --noise_type awgnc
    --snr 15
"

SC_ARGS="
    --Gamma 16
    --W 2
    --zeta 0.77
"

OUTPUT_ARGS="
    --output_path $NEW_PATH
"

# current_time=$(date +"%Y-%m-%d_%H:%M:%S")

# LOG_FILE="${LOG_PATH}run_${current_time}.log"

python3 ./decode_oamp_all.py \
    $BASE_ARGS \
    $SIGNAL_ARGS \
    $CODE_ARGS \
    $NOISE_ARGS \
    $SC_ARGS \
    $OUTPUT_ARGS
