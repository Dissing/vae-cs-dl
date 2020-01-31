from collections import namedtuple

Config = namedtuple("MonetConfig",[
    "batch_size",
    "encoder_channels",
    "decoder_channels",
    "latent_dimensions",
    "num_epochs",
    "data_dir",
    "checkpoint",
    "num_slots",
    "img_size",
])


sprites_config = Config(
    batch_size=64,
    encoder_channels=32,
    decoder_channels=32,
    latent_dimensions=16,
    num_epochs=25,
    data_dir="data/sprites/train/",
    checkpoint="checkpoints/sprites.ckpt",
    num_slots=4,
    img_size=64,
)
