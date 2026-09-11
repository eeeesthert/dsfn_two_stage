"""Print commands for the five supported ablations."""
for mode in ('sift','sift_slic','sift_slic_gabor','sift_slic_saliency','full'): print(f'python main.py --reference REF --target TGT --output outputs/{mode} --mode {mode}')
