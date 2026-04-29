#!/bin/bash
# Google Cloud GPU Training Script for DIAMOND CS2
# Usage: bash scripts/gcloud_train.sh [create|upload|train|download|delete]

set -e
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

# ─── Configuration ───
PROJECT="project-4cb9e076-b72e-4d21-8e0"
ZONE="us-central1-b"
INSTANCE="diamond-cs2-train"
MACHINE_TYPE="g2-standard-8"  # 8 vCPUs, 32GB RAM, 1x L4 GPU
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE="pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260320"
IMAGE_PROJECT="deeplearning-platform-release"

REPO_DIR="/home/cooper/mst/spring2026/cs6406/project/diamond-cs2"
REMOTE_DIR="/home/coopermw03/diamond-cs2"

# ─── Functions ───

create_vm() {
    echo "Creating GPU VM: $INSTANCE ($MACHINE_TYPE + $GPU_TYPE)"
    gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --image="$IMAGE" \
        --image-project="$IMAGE_PROJECT" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True" \
        --scopes=default,storage-rw

    echo "Waiting for VM to be ready..."
    sleep 30

    # Wait for SSH
    for i in {1..10}; do
        if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="echo ready" 2>/dev/null; then
            echo "VM is ready!"
            return 0
        fi
        echo "  Waiting for SSH... ($i/10)"
        sleep 15
    done
    echo "WARNING: VM may not be ready yet. Try again in a minute."
}

upload_code() {
    echo "Uploading code and data to VM..."

    # Create tarball of code only (exclude all data)
    cd "$REPO_DIR"
    tar czf /tmp/diamond-cs2-code.tar.gz \
        --exclude='.venv' \
        --exclude='./data' \
        --exclude='results' \
        --exclude='src/outputs' \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.flow' \
        .

    # Upload code
    gcloud compute scp /tmp/diamond-cs2-code.tar.gz \
        "$INSTANCE:~/diamond-cs2-code.tar.gz" \
        --zone="$ZONE"

    # Upload processed dataset (the important part)
    if [ -d "$REPO_DIR/data/processed" ]; then
        echo "Uploading processed dataset..."
        cd "$REPO_DIR/data"
        tar czf /tmp/diamond-cs2-data.tar.gz processed/
        gcloud compute scp /tmp/diamond-cs2-data.tar.gz \
            "$INSTANCE:~/diamond-cs2-data.tar.gz" \
            --zone="$ZONE"
    fi

    # Setup on remote
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        mkdir -p $REMOTE_DIR/data
        cd $REMOTE_DIR && tar xzf ~/diamond-cs2-code.tar.gz
        if [ -f ~/diamond-cs2-data.tar.gz ]; then
            cd $REMOTE_DIR/data && tar xzf ~/diamond-cs2-data.tar.gz
        fi
        rm -f ~/diamond-cs2-code.tar.gz ~/diamond-cs2-data.tar.gz

        # Install dependencies
        pip install h5py hydra-core omegaconf wandb tqdm torcheval opencv-python pygame 2>&1 | tail -3
        echo 'Setup complete!'
    "
}

start_training() {
    local EPOCHS=${1:-100}
    echo "Starting training for $EPOCHS epochs on $INSTANCE..."

    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        cd $REMOTE_DIR/src
        nohup python3 main.py \
            env.path_data_low_res=$REMOTE_DIR/data/processed/low_res \
            env.path_data_full_res=$REMOTE_DIR/data/processed/full_res \
            training.num_final_epochs=$EPOCHS \
            denoiser.training.batch_size=16 \
            denoiser.training.grad_acc_steps=8 \
            upsampler.training.batch_size=4 \
            upsampler.training.grad_acc_steps=8 \
            evaluation.should=True \
            evaluation.every=10 \
            > ~/training.log 2>&1 &
        echo 'Training started in background (PID: \$!)'
        echo 'Monitor with: gcloud compute ssh $INSTANCE --zone=$ZONE --command=\"tail -f ~/training.log\"'
    "
}

check_training() {
    echo "Checking training status..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        tail -30 ~/training.log 2>/dev/null || echo 'No training log found'
        echo '---'
        nvidia-smi 2>/dev/null | head -15
    "
}

download_results() {
    echo "Downloading results from VM..."
    local OUTPUT_DIR="$REPO_DIR/results/cloud_training"
    mkdir -p "$OUTPUT_DIR"

    # Find and download checkpoints
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        cd $REMOTE_DIR/src/outputs
        LATEST=\$(ls -td */ 2>/dev/null | head -1)
        if [ -n \"\$LATEST\" ]; then
            cd \"\$LATEST\"
            tar czf ~/training_results.tar.gz checkpoints/ config/ 2>/dev/null || true
            echo \"Packaged: \$LATEST\"
        else
            echo 'No training output found'
        fi
    "

    gcloud compute scp "$INSTANCE:~/training_results.tar.gz" \
        "$OUTPUT_DIR/training_results.tar.gz" \
        --zone="$ZONE" 2>/dev/null && \
        cd "$OUTPUT_DIR" && tar xzf training_results.tar.gz && \
        echo "Results downloaded to $OUTPUT_DIR" || \
        echo "No results to download yet"

    # Also grab the training log
    gcloud compute scp "$INSTANCE:~/training.log" \
        "$OUTPUT_DIR/training.log" \
        --zone="$ZONE" 2>/dev/null || true
}

delete_vm() {
    echo "Deleting VM: $INSTANCE"
    echo "Make sure you've downloaded results first!"
    read -p "Are you sure? [y/N] " confirm
    if [ "$confirm" = "y" ]; then
        gcloud compute instances delete "$INSTANCE" \
            --zone="$ZONE" \
            --quiet
        echo "VM deleted."
    fi
}

stop_vm() {
    echo "Stopping VM (preserves disk, stops billing for compute)..."
    gcloud compute instances stop "$INSTANCE" --zone="$ZONE"
    echo "VM stopped. Restart with: gcloud compute instances start $INSTANCE --zone=$ZONE"
}

estimate_cost() {
    echo "Cost estimate (L4 GPU, g2-standard-8):"
    echo "  On-demand: ~\$1.20/hr"
    echo "  100 epochs (~8 hrs): ~\$10"
    echo "  600 epochs (~48 hrs): ~\$58"
    echo "  Free credits available: \$300"
    echo ""
    echo "  TIP: Stop the VM when not training to avoid charges"
}

# ─── Main ───

case "${1:-help}" in
    create)    create_vm ;;
    upload)    upload_code ;;
    train)     start_training "${2:-100}" ;;
    check)     check_training ;;
    download)  download_results ;;
    stop)      stop_vm ;;
    delete)    delete_vm ;;
    cost)      estimate_cost ;;
    help)
        echo "Usage: bash scripts/gcloud_train.sh <command>"
        echo ""
        echo "Commands:"
        echo "  create     Create GPU VM (L4, ~\$1.20/hr)"
        echo "  upload     Upload code and dataset to VM"
        echo "  train N    Start training for N epochs (default: 100)"
        echo "  check      Check training progress"
        echo "  download   Download checkpoints and results"
        echo "  stop       Stop VM (saves disk, stops compute billing)"
        echo "  delete     Delete VM entirely"
        echo "  cost       Show cost estimates"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run 'bash scripts/gcloud_train.sh help' for usage"
        ;;
esac
