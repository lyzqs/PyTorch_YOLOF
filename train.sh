python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v fcos-r50-RT \
        --batch_size 16 \
        --schedule 3x \
        --grad_clip_norm 4.0 \
