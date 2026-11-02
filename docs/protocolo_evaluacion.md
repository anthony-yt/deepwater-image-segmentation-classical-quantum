# Protocolo de evaluación comparativa (Task 4.8)

**Estado: propuesta de Ariana, pendiente de confirmación de Dayana.**
Hasta que las dos lo acepten no hay protocolo, porque afecta igual a las 35 corridas del U-Net y
a las 70 de SegFormer y UWSegFormer.

Fecha de redacción: 2026-11-01.

---

## Para qué sirve este documento

La Task 4.8 cierra el Goal 2.4: la tabla comparativa de U-Net, SegFormer y UWSegFormer. Su
dificultad no está en entrenar, está en que los números sean comparables.

Hoy no lo son:

| | U-Net (Ariana) | SegFormer (Dayana) |
|---|---|---|
| etiquetas | 105 máscaras SAM de 3 puntos | 20 curadas de Video1 |
| frames | 84 train / 21 val | 16 train / 4 val |
| videos | 1, 3, 6 y 7, mezclados entre train y val | solo Video1 |
| framework | PyTorch puro, 512×512 | MMSegmentation |
| métrica reportada | Dice por imagen de la clase 1 | `mIoU` global de MMSeg |
| resultado | Dice 0,7485 | IoU estructura 76,0 |

Poner 0,7485 y 76,0 en la misma tabla sería inventar la comparación: no coinciden ni las
etiquetas, ni los frames, ni los videos, ni la métrica, ni cómo se promedia.

## Qué se iguala y qué no

Lo que se fija aquí son las **condiciones de medición**, no los hiperparámetros.

| Se acuerda (idéntico en los tres modelos) | Se deja libre (es lo que se compara) |
|---|---|
| qué máscaras entran al conjunto | arquitectura |
| el reparto de folds, frame por frame | learning rate, batch, optimizador |
| las semillas | augmentación propia de cada modelo |
| la resolución a la que se mide | schedule |
| la función que calcula las métricas | |
| el criterio de presupuesto y de checkpoint | |

Igualar también los hiperparámetros dejaría de comparar modelos. No igualar las condiciones de
medición hace que cualquier diferencia observada pueda venir de la medición y no del modelo.

---

## Las ocho decisiones

### 1. Qué máscaras entran al conjunto

Cuando la Task 5.1 termine habrá **68 máscaras**: 20 de Video1 y 8 de cada uno de los otros seis.

**Decisión: se usan las 68 tal cual, y la tabla final reporta el peso de cada video en el
conjunto de entrenamiento de cada fold.**

Motivo: recortar Video1 a 8 equilibraría el conjunto pero tiraría 12 máscaras ya curadas, que con
68 en total es mucho. El desbalance se documenta en vez de esconderse: `folds_4_8.meta.json`
incluye `peso_en_el_conjunto` por video.

Consecuencia que hay que tener presente al leer la tabla: en los folds que incluyen Video1, ese
video aporta cerca de un tercio de los datos de entrenamiento, así que lo que el modelo aprenda
estará sesgado hacia estructuras del tipo de Video1.

### 2. Promedio de las métricas: macro

**Decisión: macro. IoU y F1 de borde se calculan por imagen y después se promedian, primero
dentro de cada video y luego entre semillas.**

Motivo: es lo que hace el arnés compartido de la Task 5.7, y da el mismo peso a un frame con 8 %
de estructura que a uno con 95 %. La alternativa (micro: acumular TP, FP y FN de todo el video y
dividir al final) deja que los frames con mucha estructura dominen el resultado del video.

No es un detalle de implementación. El U-Net venía promediando por imagen y el `mIoU` de MMSeg
sale de una matriz de confusión acumulada: **son números distintos sobre los mismos datos**.
Aceptar esta decisión implica que las corridas de MMSegmentation no reporten su `mIoU` en la tabla
final, sino las métricas recalculadas con el arnés a partir de las máscaras predichas.

Tolerancia del F1 de borde: `tol = 2` píxeles, igual para todos. Las alturas de los frames son 360
y 352, así que 2 píxeles significa prácticamente lo mismo en todos los videos.

### 3. Presupuesto de entrenamiento y selección de checkpoint

**Decisión: presupuesto fijo de pasadas por el conjunto de entrenamiento, igual para los tres
modelos, sin selección de checkpoint.**

Motivo: en leave-one-video-out el video excluido **es el conjunto de prueba**. Elegir el mejor
checkpoint según su métrica sería seleccionar sobre el test y contaminaría el resultado. La
alternativa limpia sería apartar uno de los 6 videos de entrenamiento como validación interna,
pero con folds de 48 a 60 imágenes eso cuesta un video entero de datos.

Detalle imprescindible: **el presupuesto se expresa en pasadas por el conjunto de entrenamiento**,
no en épocas ni en iteraciones. El U-Net cuenta épocas; MMSegmentation cuenta iteraciones (las
corridas actuales usan 2000 iteraciones con `samples_per_gpu = 2`). Para un fold de N imágenes:

```
pasadas = iteraciones × samples_per_gpu / N
```

**Pendiente abierto:** el valor del presupuesto no está decidido. Sale de una corrida piloto que
debe hacerse **después** de regenerar las etiquetas. El U-Net convergía en la época 1 sobre las
máscaras SAM, pero eso era un artefacto de esas etiquetas (ver Task 4.2), así que el número no se
puede heredar.

### 4. Regla de empate

**Decisión: una diferencia entre dos modelos se reporta como real, en un video dado, solo si**

```
|media_A − media_B|  >  sd_A + sd_B
```

**donde `sd` es la desviación entre las 5 semillas de ese modelo en ese video. En caso contrario
se reporta empate.**

Si alguna diferencia queda justo en el límite, se desempata con un bootstrap emparejado por frame
cuyo intervalo del 95 % no cruce cero.

Motivo y expectativa: con 68 máscaras y folds de 48 a 60 imágenes de entrenamiento, la variación
entre semillas va a ser grande, y los modelos clásicos se separan de la línea base por unos 0,03.
**Lo más probable es que la tabla diga «no se distinguen», y eso es un resultado válido y
reportable.** Lo que no sería válido es declarar un ganador por una diferencia dentro del ruido.

Esta regla se fija **antes** de ver los resultados, a propósito.

### 5. Esquema de folds: leave-one-video-out

**Decisión: un fold por video con máscaras. En cada fold, ese video es el test y los demás el
entrenamiento. 7 folds × 5 semillas = 35 corridas por modelo.**

Motivo: es lo único que produce el resultado por video que pide el entregable, y es exactamente el
protocolo que ya define la Task 5.4 de Dayana, así que no obliga a nadie a cambiar de plan.

Un video con menos de 4 máscaras no tiene fold propio, pero sigue usándose como entrenamiento
(parámetro `--min-masks` del generador).

### 6. Semillas

**Decisión: `[0, 1, 2, 3, 4]` en los tres modelos. La semilla controla la inicialización de pesos,
el orden del dataloader y la augmentación.**

### 7. Resolución a la que se mide

**Decisión: cada modelo devuelve su predicción al tamaño original del frame, con interpolación por
vecino más cercano, y las métricas se calculan a esa resolución.**

Motivo: los frames miden 640×360, 480×360 o 640×352 según el video, así que reescalar a cuadrado
distorsiona la relación de aspecto de forma distinta en cada uno. Medir todo a 512×512 obligaría a
reescalar también la máscara de referencia, lo que inventa bordes y hace que el F1 de borde deje
de significar lo mismo entre videos.

Vecino más cercano, no bilineal: una máscara binaria interpolada suavemente deja de ser binaria.

### 8. Función de métricas

**Decisión: `evaluate()` de `src/evaluation/segmentation_metrics.py` (Task 5.7), y nada más.**

Devuelve IoU, Dice, precisión, recall y F1 de borde por frame, más la línea base «todo estructura»
y el intervalo bootstrap, y agrega entre semillas con `across_seeds()`. Dos funciones distintas no
se traducen la una a la otra a posteriori.

**Hueco conocido:** el arnés da intervalo bootstrap emparejado para *modelo contra línea base*,
pero no para *modelo contra modelo*, que es lo que la regla de empate necesita como desempate. Es
un añadido pequeño a `segmentation_metrics.py` y corresponde pedírselo a Anthony.

---

## El manifiesto de folds

Que las dos digamos «hacemos leave-one-video-out» no garantiza el mismo reparto. Lo que lo
garantiza es un único archivo con su hash.

```bash
python -m src.data.build_fold_manifest \
    --masks-dir ".../Dataset_Processed/Test_Set_Curado" \
    --out       ".../Dataset_Processed/Seleccion/folds_4_8.csv"
```

Produce:

- `folds_4_8.csv` — columnas `fold, frame_id, video, split`.
- `folds_4_8.meta.json` — `sha256` del CSV, conteos y peso por video, y la línea base «todo
  estructura» de cada video.

**Cada corrida de la Task 4.8 y de la Task 5.4 registra ese `sha256` junto a sus resultados.** Si
dos corridas no comparten hash, no son comparables y no entran en la misma tabla. El hash se puede
verificar sin ejecutar nada:

```bash
sha256sum folds_4_8.csv
```

Lo genera Anthony, que es el dueño de las etiquetas. Lo valida Dayana con el mecanismo de
`manifest_sha256` que ya usa en sus notebooks.

---

## Reparto del trabajo

- **Ariana** entrena el U-Net: 7 folds × 5 semillas = 35 corridas.
- **Dayana** no entrena para la 4.8. Reutiliza las corridas de su Task 5.4 para SegFormer y
  UWSegFormer, que son 70 bajo este mismo protocolo.

De ahí que la 4.8 «dependa de Dayana (corridas de 5.4)»: si la 5.4 corre con otro reparto u otras
semillas, esas 70 corridas no sirven y hay que repetirlas. **El acuerdo va antes de entrenar.**

## Forma del entregable

Cuatro filas por video: los tres modelos y la línea base. Cada celda con IoU y F1 de borde, media
± desviación entre semillas. La línea base no lleva desviación porque es determinista.

```
                      Video1            Video2       ...      Video6
modelo             IoU       bF1       IoU     bF1          IoU      bF1
U-Net            .___±.__  .__±.__                        .__±.__
SegFormer        .___±.__  .__±.__                        .__±.__
UWSegFormer      .___±.__  .__±.__                        .__±.__
todo estructura   0.718       —                            0.081     —
```

La fila de la línea base es la que da sentido al resto: **0,718 en Video1 y 0,081 en Video6**. Un
IoU de 0,70 es un empate con no hacer nada en Video1 y un resultado excelente en Video6.

Corolario que afecta a otras tasks: **cualquier umbral absoluto de mejora cambia de significado
según el video.** El criterio de 2,0 puntos de IoU del documento de diseño híbrido (Task 5.14) está
fijado contra una línea base supuesta de 0,72, que es la de Video1 y de ningún otro.

---

## Precondiciones

| Precondición | Estado al 2026-11-01 | Responsable |
|---|---|---|
| Máscaras curadas de la Task 5.1 | **28 de 68** (Video1 ×20, Video6 ×8) | Anthony |
| Arnés de la Task 5.7 | hecho, en `feat/labelling-evaluation-tools`, sin mergear | Anthony |
| `mmseg_pipes` regenerado desde `Test_Set_Curado` | no hecho | Anthony |
| Manifiesto de folds con `sha256` | generador listo, falta ejecutarlo con las 68 | Anthony |
| Corridas de la Task 5.4 | no empezadas | Dayana |
| Presupuesto de pasadas, vía corrida piloto | no decidido | Ariana y Dayana |
| Bootstrap modelo contra modelo en el arnés | no hecho | Anthony |

La cuarta precondición no es menor: `mmseg_pipes` se construye hoy desde
`SAM_Masks_Frames_CLAHE`, y la Task 4.3 mide esas máscaras contra el ground truth curado:
IoU 0,664 en Video1 frente a una base de 0,718, e **IoU 0,083 en Video6, donde la máscara marca el
95 % del frame mientras la estructura real ocupa el 8 %**. Entrenar la 4.8 sobre esas etiquetas
compararía tres modelos sobre ruido.

---

## Referencias

- `notebooks/3_2_unet_training_diagnosis.ipynb` — Task 4.2, por qué el U-Net iguala la predicción
  constante y por qué el split actual no mide generalización.
- `notebooks/2_3_sam_mask_audit.ipynb` — Task 4.3, estado de las máscaras SAM de 3 puntos.
- `src/evaluation/segmentation_metrics.py` — Task 5.7, el arnés.
- `src/data/build_fold_manifest.py` — generador del manifiesto descrito arriba.
- `docs/taxonomia_errores_5_13.md` — taxonomía de causas para el análisis de errores conjunto.
