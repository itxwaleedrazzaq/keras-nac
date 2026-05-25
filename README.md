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
    d_model=64,                 # Dimension of the model of LAN
    num_heads=16,               # Number of attention heads of LAN
    topk=8,                     # Number of top-k attention interactions
    euler_steps=6,              # Number of Euler steps 
    activation="sigmoid",       # Activation function
    use_sink_gate=True,         # Use Attention Sink Gate
    return_sequences=False,     # Return full sequences if True, else last output
    return_attention=False      # Return attention weights if True
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
    d_model=64,                 # Dimension of the model of LAN
    num_heads=16,               # Number of attention heads of LAN
    num_layers=1,               # Number of stacked encoder/decoder layers
    ff_dim=32,                  # Dimension of the feed-forward network
    delta_t= 0.01,              # Time-step for the Liquid Attention
    euler_steps=5,              # Number of Euler steps for Liquid Attention
    topk=8,                     # Number of top-k attention interactions
    expansion_rate=2,           # Expansion factor for feed-forward layers
    use_sink_gate=True,         # Enable sink gate mechanism
    use_pairwise=False,         # disable top-k sparsity if True
    enable_hc=True,             # Enable hyper-connections if True, Otherwise -> Residual connections
    dynamic_hc=True,            # Enable Liquid hyper-connections if True, Otherwise -> Static
    dropout=0.0,                # Dropout rate
    max_len=1000,               # Maximum sequence length of positional encoder
    return_attention=False,     # Return attention weights if True
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
    d_model=64,                  # Dimension of the model
    num_heads=16,                # Number of attention heads
    mode='exact',                # Computation mode: 'exact', 'euler', or 'steady'
    topk=8,                      # Number of top-k pairwise interactions
    delta_t=0.5,                 # Time step for Euler mode
    sparsity=0.5,                # Sparsity level for NCP wiring
    euler_steps=6,               # Number of Euler integration steps
    dropout=0.0,                 # Dropout rate
    tau_epsilon=1e-5,            # Small positive value for temporal head
    activation='sigmoid',        # Activation function
    use_riemann_sum=True         # Use Reimann-sum integration if True, else standard weighted sum
    return_sequences=False,      # Return full sequences if True, else last output
    return_attention=False,      # Return attention weights if True
    return_cell_state=False      # Return cell-level state  if True
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
    mean, log_std = OUWrap(
        NAC(d_model=64, num_heads=16, topk=8, sparsity=0.5),
        output_dim=1,                # Output dimension for regression 
        bn_mean=0.0,                 # Brownian mean
        bn_std=0.1,                  # Brownian standard deviation
        activation='sigmoid',        # Activation function
        return_sequences=False,      # Return full sequences if True, else last output
        return_attention=False,      # Return attention weights if True
        return_cell_state=False,     # Return cell potentials if True
        )(inputs)
    return tf.keras.Model(inputs, [mean, log_std])

model = NSAC(stochastic_model_fn(),
             mc_samples=5,           # Monte-Carlo steps 
             ood_mean=0.0,           # OOD generating noise mean 
             ood_std=5.0             # OOD generating noise standard deviation
             )

model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=NSACLoss(),
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

