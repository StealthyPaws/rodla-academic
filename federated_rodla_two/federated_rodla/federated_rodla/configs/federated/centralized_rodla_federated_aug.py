# configs/federated/centralized_rodla_federated_aug.py

_base_ = '../../rodla_internimage_xl_publaynet.py'  # CHANGED to PubLayNet

# Federated data settings for PubLayNet-P
federated_data = dict(
    server_url='localhost:8080',
    client_id='client_01',
    data_batch_size=50,
    max_samples_per_epoch=1000,
    perturbation_types=[
        'background', 'defocus', 'illumination', 'ink_bleeding', 'ink_holdout',
        'keystoning', 'rotation', 'speckle', 'texture', 'vibration', 
        'warping', 'watermark', 'random', 'all'
    ],
    severity_levels=[1, 2, 3]  # CHANGED: Discrete levels instead of privacy levels
)

# Training remains exactly the same
# The only change: we'll modify the data loader to use federated augmented data