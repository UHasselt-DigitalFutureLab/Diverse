import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class FiLMLayer(tf.keras.layers.Layer):
    """
    ----------------------------------------------------------------------------
    Frozen FiLM layer (Feature-wise Linear Modulation) *without* trainable
    parameters.  Suitable for evolutionary search where only the latent vector
    ``z`` changes (e.g. via CMA-ES).

        γ = 1 + tanh( z @ W_gamma )
        β =         tanh( z @ W_beta )

    *  ``units``         : number of channels / neurons to be modulated.
    *  ``projection_dim``: length of the latent vector z  ( ≪ units is OK ).
    *  ``W_gamma``, ``W_beta`` : frozen random projection matrices.
    *  ``init_std``      : std-dev of the frozen weights. Raising it increases
                           the raw variance *before* tanh, giving CMA-ES a
                           stronger lever.
    ----------------------------------------------------------------------------
    """
    def __init__(self, units, projection_dim,
                 init_std=0.5, random_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.projection_dim = projection_dim
        self.init_std = init_std  # variance of the initial random frozen weights

        # Create the frozen random projection matrices
        self.W_gamma = self.add_weight(
            shape=(projection_dim, units),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.init_std, seed=random_seed),
            trainable=False,
            name='W_gamma'
        )
        self.W_beta = self.add_weight(
            shape=(projection_dim, units),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.init_std, seed=random_seed + 1),
            trainable=False,
            name='W_beta'
        )

    def call(self, inputs: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
         # 1. Linear projection  z → raw (unclamped)  γ  and  β
        raw_gamma = tf.matmul(z, self.W_gamma)
        raw_beta  = tf.matmul(z, self.W_beta)

        # 2. Tanh clamp to keep the range safe and bounded
        gamma = 1.0 + tf.tanh(raw_gamma)
        beta  = tf.tanh(raw_beta)

        # 3. Apply FiLM modulation
        #    • For 2-D inputs  (B,C)  ➞ broadcasting works automatically.
        #    • For 4-D inputs (B,H,W,C) we need to add spatial dims to γ,β.
        if inputs.shape.rank == 2:          # (B, C)
            return gamma * inputs + beta
        elif inputs.shape.rank == 4:                               # (B, H, W, C)
            # Expand γ, β to match the input shape (B, H, W, C)
            # -1 means "keep the original size for that dimension"
            # 1 means "expand this dimension to size 1"
            gamma = tf.reshape(gamma, (-1, 1, 1, self.units))
            beta  = tf.reshape(beta,  (-1, 1, 1, self.units))
            return gamma * inputs + beta
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}. Expected 2D or 4D tensor.")

    def get_gamma_beta(self, z: tf.Tensor):
            """Return (γ, β) for a given latent vector z, for debugging."""
            raw_gamma = tf.matmul(z, self.W_gamma)
            raw_beta  = tf.matmul(z, self.W_beta)
            gamma = 1.0 + tf.tanh(raw_gamma)
            beta  = tf.tanh(raw_beta)
            return gamma, beta