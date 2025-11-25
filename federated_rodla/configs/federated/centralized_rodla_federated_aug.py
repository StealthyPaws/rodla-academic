# configs/federated/centralized_rodla_federated_aug.py

_base_ = '../../rodla_internimage_xl_m6doc.py'

# Keep original RoDLA model COMPLETELY UNCHANGED
# We only modify the data source for training

# Federated data settings
federated_data = dict(
    server_url='localhost:8080',
    client_id='client_01',
    data_batch_size=50,  # Number of samples to send per batch
    max_samples_per_epoch=1000,  # Limit samples per epoch
    privacy_level='medium',  # low/medium/high
    augmentation_types=['geometric', 'color', 'noise', 'blur']
)

# Training remains exactly the same
# The only change: we'll modify the data loader to use federated augmented data