## keras-nac

This repository contains TensorFlow/Keras implementations of the following research papers:

- FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning
- Neuronal Attention Circuit (NAC) for Representation Learning
- Neuronal Stochastic Attention Circuit (NSAC) for Probabilistic Representation Learning
---

### Installation

```bash
pip install keras-nac
```

---

### Requirements

- Python >= 3.10
- TensorFlow >= 2.18.0

---

## Usage Examples [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tww65dcYYFDKjx49cTh0u4D8IfqJhVst?usp=sharing)

These layers can be used as drop-in components inside TensorFlow/Keras models.

---

### 1. Liquid Attention Network (LAN)

```python
import tensorflow as tf
from keras_nac import layers

inputs = tf.keras.Input(shape=(1, 1))

attn = layers.LAN(
    d_model=64,
    num_heads=16,
    topk=32,
    euler_steps=6,
    activation="sigmoid",
    use_sink_gate=True,
    return_sequences=False,
    return_attention=False
)(inputs)

outputs = tf.keras.layers.Dense(2)(attn)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

### 2. FLUID Transformer

```python
import tensorflow as tf
from keras_nac import layers

inputs = tf.keras.Input(shape=(1, 1))

x = layers.FLUID(
    d_model=64,
    num_heads=16,
    num_layers=1,
    ff_dim=32,
    delta_t=0.01,
    euler_steps=5,
    topk=8,
    expansion_rate=2,
    use_sink_gate=True,
    use_pairwise=False,
    enable_hc=True,
    dynamic_hc=True,
    dropout=0.0,
    max_len=1000,
    return_attention=False
)(inputs)

x = tf.keras.layers.Activation("sigmoid")(x)
x = tf.keras.layers.Flatten()(x)

outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

### 3. Neuronal Attention Circuit (NAC)

```python
import tensorflow as tf
from keras_nac import layers

inputs = tf.keras.Input(shape=(1, 1))

x = layers.NAC(
    d_model=64,
    num_heads=16,
    mode="exact",
    topk=8,
    delta_t=0.5,
    sparsity=0.5,
    euler_steps=6,
    dropout=0.0,
    tau_epsilon=1e-5,
    activation="sigmoid",
    use_riemann_sum=True,
    return_sequences=False,
    return_attention=False,
    return_cell_state=False
)(inputs)

outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

### 4. Neuronal Stochastic Attention Circuit (NSAC)

```python
import tensorflow as tf
from keras_nac import layers, models, losses


def stochastic_model_fn():
    inputs = tf.keras.Input(shape=(1, 1))

    mean, log_std = layers.OUWrap(
        layers.NAC(
            d_model=64,
            num_heads=16,
            topk=8,
            sparsity=0.5
        ),
        output_dim=1,
        bn_mean=0.0,
        bn_std=0.1,
        activation="sigmoid",
        return_sequences=False,
        return_attention=False,
        return_cell_state=False
    )(inputs)

    return tf.keras.Model(inputs, [mean, log_std])


model = models.NSAC(
    stochastic_model_fn(),
    mc_samples=5,
    ood_mean=0.0,
    ood_std=5.0
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=losses.NSACLoss(lambda_reg=0.5),
)
```

---
## Citation

```bibtex
@article{razzaq2025neuronal,
  title={Neuronal Attention Circuit (NAC) for Representation Learning},
  author={Razzaq, Waleed and Kanjaraway, Izis and Zhao, Yun-Bo},
  journal={arXiv preprint arXiv:2512.10282},
  year={2025}
}

@article{razzaq2026fluid,
  title={FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning},
  author={Razzaq, Waleed and Zhao, Yun-Bo},
  journal={arXiv preprint arXiv:2605.04421},
  year={2026}
}
---
