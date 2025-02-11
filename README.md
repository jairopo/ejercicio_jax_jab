# Eliminación de Ruido en Imágenes con JAX

> Jairo Andrades Bueno

## Descripción

Este proyecto implementa un autoencoder utilizando JAX para eliminar el ruido en imágenes del dataset CIFAR-10. Se entrena una red neuronal convolucional para aprender a reconstruir imágenes ruidosas y restaurarlas a su versión original.

## Contenido del Repositorio

- [Informe_JAX.pdf](Informe_JAX.pdf): Información sobre JAX sobre sus características, ecosistema y comparación con PyTorch y TensorFlow.

- [ejercicio_JAX_Jairo_Andrades_Bueno.ipynb](ejercicio_JAX_Jairo_Andrades_Bueno.ipynb): Código principal que incluye la carga de datos, preprocesamiento, definición del modelo, entrenamiento y visualización de resultados.

## Estructura del Código

### 1. Importación de Librerías
El código comienza con la importación de las librerías necesarias:
- `jax`, `jax.numpy`: operaciones en tensores y computación acelerada.
- `optax`: optimización del modelo.
- `flax.linen`: definir el autoencoder.
- `numpy`: manipulación de datos.
- `matplotlib.pyplot`: visualización de resultados.
- `tensorflow.keras.datasets`: carga del dataset CIFAR-10.

### 2. Carga y Normalización del Dataset
Se carga el conjunto de datos CIFAR-10 y se normaliza dividiendo los valores de los píxeles por 255.0 para que estén en el rango `[0, 1]`.
```python
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
```

### 3. Función para Agregar Ruido
Se define `add_random_noise`, que introduce ruido aleatorio en las imágenes:
```python
def add_random_noise(images, num_points):
    noisy_images = images.copy()
    for img in noisy_images:
        height, width, _ = img.shape
        for i in range(num_points):
            x = np.random.randint(0, height)
            y = np.random.randint(0, width)
            img[x, y] = np.random.uniform(0, 1, size=3)
    return noisy_images
```
Luego, se aplica esta función a los conjuntos de entrenamiento y prueba.

### 4. Definición del Autoencoder
Se define el modelo `Model` usando Flax:
- **Encoder:** Dos capas convolucionales seguidas de max-pooling.
- **Decoder:** Dos capas convolucionales y dos operaciones de upsampling (reescalado bilineal).
- **Salida:** Capa convolucional con activación sigmoide para obtener valores entre `[0,1]`.
```python
class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = nn.max_pool(x, (2, 2), padding='SAME')
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = nn.max_pool(x, (2, 2), padding='SAME')
        
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = jax.image.resize(x, (x.shape[0], 16, 16, 32), method='bilinear')
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME')(x))
        x = jax.image.resize(x, (x.shape[0], 32, 32, 32), method='bilinear')
        x = nn.sigmoid(nn.Conv(3, (3, 3), padding='SAME')(x))
        return x
```

### 5. Inicialización del Modelo y Estado de Entrenamiento
Se inicializan los parámetros del modelo y el optimizador Adam:
```python
def create_train_state(rng, learning_rate=1e-3):
    model = Model()
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
```

### 6. Función de Pérdida y Optimización
Se define la función de pérdida basada en el error cuadrático medio:
```python
def loss_fn(params, batch):
    recon = Model().apply(params, batch)
    return jnp.mean((recon - batch) ** 2)
```

Se optimizan los parámetros con `jax.jit` para mayor eficiencia:
```python
@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss
```

### 7. Entrenamiento del Modelo
El modelo se entrena por 10 épocas usando un batch size de 128:
```python
batch_size = 128
epochs = 10
for epoch in range(epochs):
    loss_accum = 0.0
    for i in range(0, 40000, batch_size):
        batch = x_train_noisy[i:i+batch_size]
        state, loss = train_step(state, batch)
        loss_accum += loss
    print(f'Epoch {epoch+1}, Loss: {loss_accum / (40000 / batch_size)}')
```

### 8. Restauración de Imágenes Ruidosas
Se usa el modelo entrenado para denoizar imágenes de prueba:
```python
def denoise_images(state, images):
    return Model().apply(state.params, images)

decoded_images = denoise_images(state, x_test_noisy[:5])
```

### 9. Visualización de Resultados
Se muestran imágenes originales, con ruido y restauradas.
```python
plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(3, 5, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.title("Original")

    plt.subplot(3, 5, i+6)
    plt.imshow(x_test_noisy[i])
    plt.axis('off')
    plt.title("Con ruido")

    plt.subplot(3, 5, i+11)
    plt.imshow(decoded_images[i])
    plt.axis('off')
    plt.title("Restaurada")
plt.show()
```

## Recursos Utilizados
- <https://jax.readthedocs.io/en/latest/quickstart.html>

- <https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/examination-review-introduction-to-jax/what-is-jax-and-how-does-it-speed-up-machine-learning-tasks/>

- <https://geekflare.com/es/google-jax/>

- <https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/examination-review-introduction-to-jax/what-are-the-features-of-jax-that-allow-for-maximum-performance-in-the-python-environment/>

- <https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html>

- <https://www.educative.io/answers/what-is-the-jax-ecosystem>

- <https://www.kaggle.com/code/aakashnain/building-models-in-jax-part1-stax>

- <https://wandb.ai/wandb_fc/tips/reports/How-To-Create-an-Image-Classification-Model-in-JAX-Flax--VmlldzoyMjA0Mjk1>


