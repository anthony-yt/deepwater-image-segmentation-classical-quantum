# Task 4.7: diseño de arquitectura híbrida dual-encoder

## 1. Decisión de diseño

La propuesta para Sprint 5 conserva UWSegFormer como ruta principal y añade una ruta ligera de contexto. La integración se hace fuera de los encoders, en un nodo de fusión con interfaces explícitas. Esto evita reescribir el backbone, el cargador de checkpoints y el pipeline de MMSegmentation.

La hipótesis que debe probarse es concreta:

> Una segunda ruta aporta valor si mejora la segmentación de la estructura y sus bordes frente a UWSegFormer solo, en el mismo split, sin pagar un costo desproporcionado en latencia o memoria.

No se recomienda implementar CDGLU, PCSA, ACAF, la fusión tipo HRNet y priors SAM en un solo cambio. La primera versión debe aislar el valor de la segunda ruta y del mecanismo de fusión. Los módulos restantes entran por ablaciones.

## 2. Evidencia usada

- Hosseini et al. proponen un encoder local basado en EfficientNet-B0 y un encoder global Swin-T simplificado. Alinean sus salidas a `32 × 32 × 256`, aplican cross-attention de tres cabezas y refinan con CBAM antes de decodificar [1, §§3.2-3.5]. Su ablación pasa de 87.5% mIoU para la rama Swin-T y 85.2% para la rama EFS-Net a 92.8% al combinarlas con una cabeza. Tres cabezas alcanzan 96.3% y el modelo completo reporta 98.44% [1, Tabla 3]. Esos números pertenecen a HOMOMO y no deben trasladarse al dataset del proyecto.
- La presentación del proyecto resume el conflicto entre detalle local y trayectoria global, propone cross-attention de tres cabezas y sugiere ramas paralelas de alta resolución [2, diapositivas 3-5]. También plantea priors SAM para separar primer plano y fondo [2, diapositiva 6].
- DMSNet combina CDGLU, PCSA y ACAF. PCSA refuerza rasgos locales y globales, mientras ACAF conecta el rasgo global con rasgos espaciales locales del decoder [3]. Por atribución correcta, ACAF pertenece a DMSNet, no a SP-UIQA.
- SP-UIQA utiliza máscaras suaves de SAM generadas offline, fusión multiescala guiada por máscara y cross-attention espacial para modelar regiones [4]. El artículo es de evaluación de calidad, no de segmentación. Aquí solo inspira el protocolo de priors y robustez.
- SegFormer MiT-B0 entrega cuatro escalas y un decoder MLP liviano. El baseline de `notebooks/5_segformer_mit_b0_baseline.ipynb` fija la comparación sin componentes especializados [5].

### 2.1 Resultado experimental provisional de Task 4.7

La comparación se realizó sobre las mismas cuatro imágenes de validación y el mismo manifest (SHA-256 `d279983b1ca63e8b4b54f49e008fadf94f84b17c65bc201d875ddae68fb980dc`). Se evaluó el checkpoint disponible con mejor mIoU de cada experimento; para SegFormer fue `iter_1500.pth`.

| Métrica | SegFormer MiT-B0 | UWSegFormer | Delta SegFormer - UWS |
|---|---:|---:|---:|
| IoU estructura | 0.767036 | 0.763267 | +0.003768 |
| Dice estructura | 0.868161 | 0.865742 | +0.002419 |
| Recall estructura | 0.767168 | 0.763284 | +0.003883 |
| Especificidad de fondo | 0.999533 | 0.999940 | -0.000407 |
| Boundary-F1@2px | 0.027464 | 0.021066 | +0.006398 |
| Latencia mediana (ms) | 50.486 | 55.492 | -5.006 |
| Parámetros (M) | 3.715 | 16.814 | -13.099 |

En este split provisional, SegFormer obtiene una ventaja pequeña en IoU (+0.38 puntos porcentuales), Dice (+0.24 puntos) y recall (+0.39 puntos), además de una mejora mayor en Boundary-F1@2px (+0.64 puntos). También reduce la latencia mediana aproximadamente 9.0% y usa 77.9% menos parámetros. UWSegFormer conserva una ventaja marginal en precisión y especificidad de fondo, inferior a 0.05 puntos porcentuales.

Por tanto, la complejidad adicional de UWSegFormer no queda justificada por esta evaluación. Esto no demuestra superioridad general de SegFormer: `n=4` es insuficiente para inferencia estadística y el ground truth es provisional. Task 4.8 debe repetir la comparación con las anotaciones finales y agrupación por video. Para Sprint 5, SegFormer pasa a ser también un control de eficiencia obligatorio: la arquitectura híbrida debe compararse tanto contra UWSegFormer solo como contra este baseline ligero.

## 3. Arquitectura propuesta para Sprint 5

### 3.1 Entradas y tensores

Entrada RGB CLAHE: `B × 3 × 480 × 640`.

Ruta A, UWSegFormer existente:

| Nivel | Resolución | Canales esperados | Uso |
|---|---:|---:|---|
| A1 | 1/4 | 32 | bordes y textura |
| A2 | 1/8 | 64 | forma local |
| A3 | 1/16 | 160 | semántica intermedia |
| A4 | 1/32 | 256 | contexto global de UWSegFormer |

Ruta B, contexto ligero:

1. `PatchEmbed`, kernel 4, stride 4, 64 canales.
2. Bloque contextual B1 a 1/4, dos bloques de atención por ventanas.
3. `PatchMerging` a 1/8, 128 canales, dos bloques.
4. `PatchMerging` a 1/16, 256 canales, dos bloques.
5. No se implementa una cuarta etapa en la primera versión. Hosseini et al. retienen tres etapas para preservar estructuras lineales y reducir costo [1, §3.3].

Si reutilizar Swin-T introduce incompatibilidades con el stack legacy, la alternativa mínima es un `ContextLiteEncoder` convolucional con depthwise separable convolutions y dilataciones `1, 2, 4`. La interfaz de salida se mantiene en `B × 256 × 30 × 40`, por lo que no cambia el fusionador.

### 3.2 Alineación y fusión

Primera implementación, un solo nodo a 1/16:

1. Proyectar `A3` de 160 a 256 canales con `Conv1×1 + BN + GELU`.
2. Proyectar `B3` de 256 a 256 canales con `Conv1×1 + BN + GELU`.
3. Usar A3 como `Q` y B3 como `K,V` en cross-attention de tres cabezas.
4. Sumar residual de A3.
5. Aplicar `Conv3×3 + BN + GELU`.
6. Entregar `F3 = B × 256 × 30 × 40` al decoder.

Se fija la dirección `Q = UWSegFormer`, `K,V = contexto` porque el objetivo es condicionar los rasgos ya útiles de UWSegFormer, no reemplazarlos. La dirección inversa queda como ablación.

La atención completa sobre 1,200 tokens es viable, pero debe medirse. Si consume demasiada memoria, se reduce `K,V` con pooling 2×2 y se conserva `Q` a 1/16. No se baja de 1/16 en la primera versión porque las tuberías delgadas perderían continuidad.

### 3.3 Rama de alta resolución tipo HRNet

La ruta A1 se mantiene viva a 1/4 hasta el decoder. Antes de cada bloque de reconstrucción:

- `F3` sube a 1/8 y se fusiona con A2 proyectado.
- La salida sube a 1/4 y se fusiona con A1 proyectado.
- Cada fusión usa suma ponderada aprendible, seguida de `Conv3×3`.

Esto adopta la idea de intercambio multirresolución de la diapositiva 5 sin replicar HRNet completo. El cambio queda localizado en el decoder y conserva las salidas actuales del encoder.

### 3.4 Decoder y módulos candidatos

Versión mínima:

1. `F3`: `Conv3×3`, 256 a 128.
2. Upsample bilinear 2×, fusión con A2, `Conv3×3`, 128 a 64.
3. Upsample bilinear 2×, fusión con A1, `Conv3×3`, 64 a 32.
4. Upsample 4× a la resolución de entrada.
5. `Conv1×1`, 32 a 2 logits.

Ablaciones posteriores:

- **CDGLU** en la ruta de contexto, antes de B3. Su pregunta experimental es si mejora invariancia ante brillo y escala.
- **PCSA** sobre F3. Su pregunta es si mejora IoU de estructura sin elevar falsos positivos de fondo.
- **ACAF** en los dos skip connections del decoder. Sustituye la suma ponderada por atención guiada por el rasgo global. DMSNet usa tres módulos ACAF; aquí se empieza con dos porque solo se conservan A1 y A2 [3].
- **Fusión multiescala guiada por máscara**, inspirada en SP-UIQA. El nombre descriptivo se usa deliberadamente: la fuente consultada no confirma `SMMF` como acrónimo canónico [4].

### 3.5 Pérdida

Primera versión:

`L = 0.5 × CrossEntropy + 0.5 × DiceLoss`.

Después de estabilizar la implementación, añadir `BoundaryLoss` con peso 0.1 y renormalizar los otros pesos. No se cambia loss y arquitectura en la misma ablación.

## 4. Viabilidad sobre el código actual

El cambio puede limitarse a tres clases nuevas:

```text
mmseg/models/backbones/context_lite_encoder.py
mmseg/models/necks/dual_encoder_cross_attention.py
mmseg/models/decode_heads/acaf_segformer_head.py
```

Y a un wrapper:

```text
DualEncoderSegmentor
  primary_backbone = UWSegFormer.backbone
  context_backbone = ContextLiteEncoder o SimplifiedSwin
  neck = DualEncoderCrossAttention
  decode_head = HRFusionHead, luego ACAFHead por ablación
```

Se reutilizan sin cambios:

- `CustomDataset`, pipelines y manifest;
- checkpoint de UWSegFormer para la ruta A;
- runner, hooks, logging y evaluación;
- pesos del head actual como referencia, pero no se cargan si cambian sus dimensiones.

Riesgos de implementación:

| Riesgo | Prueba temprana | Mitigación |
|---|---|---|
| Formas incompatibles | test unitario con entrada `2×3×480×640` | proyectores 1×1 y asserts de shape |
| OOM por cross-attention | forward y backward en T4, batch 2 | pooling de K,V, AMP o batch 1 con acumulación |
| Segunda rama redundante | similitud CKA/coseno entre A3 y B3 | reducir la rama o cancelar el diseño |
| Pérdida de bordes | Boundary F1 y recall en objetos delgados | conservar A1 y añadir ACAF solo si aporta |
| Sobreajuste por pocos datos | tres seeds y bootstrap por imagen/grupo | congelar ruta A al inicio y regularizar ruta B |

## 5. Plan de ablación

Todas las filas usan el mismo manifest, ground truth, augmentations, número de iteraciones y seeds `{17, 42, 73}`.

| ID | Cambio respecto a UWSegFormer | Pregunta |
|---|---|---|
| A0 | UWSegFormer solo | referencia obligatoria |
| A1 | + ruta B + concatenación | ¿la segunda rama contiene señal útil? |
| A2 | + cross-attention 1 cabeza | ¿la fusión adaptativa supera concatenación? |
| A3 | cross-attention 3 cabezas | ¿tres subespacios justifican el costo? |
| A4 | + rama A1 de alta resolución | ¿mejoran bordes y objetos delgados? |
| A5 | + PCSA | ¿mejora la separación sin fuga al fondo? |
| A6 | + ACAF en skips | ¿los rasgos globales filtran ruido local? |
| A7 | + CDGLU | ¿aumenta robustez a brillo y escala? |
| A8 | + prior SAM offline | ¿el prior ayuda sin volver frágil al modelo? |

Se detiene una línea si no mejora el endpoint principal o si viola el límite de eficiencia. No se interpreta A8 antes de cerrar A0-A7.

## 6. Métrica que confirma que la fusión vale la pena

Endpoint primario:

- `Δ IoU_structure >= +2.0` puntos porcentuales frente a UWSegFormer.
- El límite inferior del intervalo bootstrap 95%, remuestreado por video/grupo, debe ser mayor que 0.
- La variante también debe superar el IoU de SegFormer MiT-B0 bajo el mismo protocolo; superar únicamente a UWSegFormer no basta, porque SegFormer fue el mejor baseline provisional y es mucho más eficiente.

Condiciones simultáneas:

- `Δ Boundary-F1@2px >= +2.0` puntos.
- La especificidad de fondo no puede caer más de 0.5 puntos.
- El recall de estructura no puede caer más de 1 punto.
- Mediana de latencia no aumenta más de 25% en el mismo GPU y resolución.
- Parámetros no aumentan más de 35% frente a UWSegFormer.

Con el split provisional actual, estos umbrales son criterios de decisión, no evidencia estadística suficiente. Task 4.8 debe repetir la comparación con el ground truth final y grupos independientes.

También se reportan mIoU y mDice para continuidad, pero no deciden por sí solos. En una clase de fondo dominante pueden esconder omisiones del objetivo.

## 7. Validación de separación fondo/objetivo inspirada en SAM

SAM no se ejecuta en este sprint. Sprint 5 debe guardar sus máscaras como artefactos offline, nunca generarlas dentro del `Dataset.__getitem__`.

### 7.1 Métricas base

- IoU, Dice, precisión y recall de `structure`.
- Especificidad de fondo y `FPR_background = FP / (FP + TN)`.
- Boundary-F1 con tolerancias de 1, 2 y 4 píxeles.
- Hausdorff 95 y distancia media simétrica de superficie.
- Tasa de componentes espurios fuera del objetivo y tamaño del mayor falso positivo.

### 7.2 Consistencia con el prior

Para una máscara SAM `P`, predicción `Y` y ground truth `G`:

- `leak_outside_prior = |Y ∩ ¬P| / |Y|`.
- `miss_inside_prior = |G ∩ P ∩ ¬Y| / |G ∩ P|`.
- `recovery_outside_prior = |Y ∩ G ∩ ¬P| / |G ∩ ¬P|`.

La tercera métrica evita premiar a un modelo que copie SAM. Si SAM omite una parte correcta, el modelo debe poder recuperarla.

### 7.3 Prueba de perturbación del prior

Evaluar el mismo checkpoint con:

1. prior correcto;
2. erosión y dilatación de 3 y 7 píxeles;
3. traslación de 5% del ancho/alto;
4. eliminación aleatoria de 25% y 50% de instancias;
5. prior vacío.

Criterio de robustez: con perturbación moderada, IoU de estructura no cae más de 3 puntos; con prior vacío, el modelo no cae más de 5 puntos respecto a la variante sin prior. Si falla, el prior funciona como atajo y no como guía.

### 7.4 Estratos obligatorios

Reportar métricas por:

- proporción de foreground;
- IoU entre SAM y ground truth en terciles;
- nivel de brillo/contraste;
- oclusión visible;
- video de origen.

El resultado global debe acompañarse de intervalos bootstrap por video. No se mezclan frames correlacionados como observaciones independientes.

## 8. Entregables de Sprint 5

- config versionado de A0-A8;
- checkpoints y SHA-256 del manifest;
- `metrics_per_image.csv` con grupo de video;
- tabla de ablación con media, desviación entre seeds e intervalo bootstrap;
- perfiles de parámetros, VRAM, latencia mediana y p95;
- panel cualitativo fijo con éxitos, falsos positivos, omisiones y casos de prior SAM incorrecto.
