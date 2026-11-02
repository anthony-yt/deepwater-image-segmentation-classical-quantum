# Taxonomía de causas de error (Task 5.13)

**Estado: propuesta de Ariana, pendiente de acordar con Dayana.**

La Task 5.13 pide que las dos definamos una lista común de causas **antes** de clasificar nada. Si
cada una inventa la suya al ver sus predicciones, las dos tablas no se pueden unir y el entregable
no responde a la pregunta de si los dos modelos fallan en los mismos frames.

Este documento es la lista inicial para discutir. No hace falta que sea perfecta; hace falta que
sea la misma para las dos y que esté escrita antes de mirar los resultados.

---

## Procedimiento

1. Las predicciones salen de la Task 4.8 (U-Net, Ariana) y de la Task 5.4 (SegFormer, Dayana),
   bajo el protocolo de `docs/protocolo_evaluacion.md`.
2. Por cada modelo y cada video: **los 10 frames con peor IoU**, con la máscara de referencia
   superpuesta.
3. A cada caso se le asigna **una causa primaria y, opcionalmente, una secundaria**. Nunca más de
   dos, y la primaria es la que explica la mayor parte del error.
4. La clasificación se guarda en Drive como
   `Registro_Etiquetado/errores_5_13_<modelo>.csv`, con columnas
   `modelo, video, frame_id, iou, causa_primaria, causa_secundaria, nota`.
5. Los indicadores numéricos de cada causa sirven para **priorizar y justificar**, no para decidir
   automáticamente. La asignación es visual.

---

## Las causas

### `visibilidad_baja`
Turbidez, dispersión o falta de contraste que impide ver dónde acaba la estructura. El error es de
percepción, no de forma.

Indicador orientativo: en `Seleccion/seleccion_5_1.csv`, los frames marcados `tipo = baja_nitidez`
y los de `contraste` bajo.

**Aviso importante sobre el umbral:** la varianza laplaciana **no es comparable entre videos**. En
Video3 va de 2,8 a 206; en Video6 de 359 a 527; en Video7 de 226 a 327. Los frames que Anthony
marcó como `baja_nitidez` en Video6 tienen varianza 359 y 388, más alta que *cualquier* frame de
Video3. Así que la nitidez baja se juzga **relativa a su propio video**, no contra un valor
absoluto.

### `estructura_pequena`
La estructura ocupa una fracción pequeña del frame y el modelo la pierde dentro del fondo.

Indicador: fracción de estructura en la máscara de referencia por debajo de 0,15.

Esta causa es casi todo Video6: sus 8 máscaras curadas van de 0,040 a 0,127 de estructura, frente
al 0,718 medio de Video1. Es un régimen distinto, no un caso difícil del mismo régimen.

### `estructura_parcial`
La estructura entra y sale del frame, o está ocluida, y el modelo segmenta solo el trozo
claramente visible o extiende la máscara por donde el objeto ya no está.

### `sobresegmentacion_global`
El modelo marca casi todo el frame como estructura.

Indicador: fracción predicha por encima de 0,90 cuando la de referencia está por debajo de 0,30.

Es el modo de fallo que ya se observó en las máscaras SAM de 3 puntos sobre Video6 (95 % predicho
frente a 8 % real, precisión 0,083). Merece categoría propia porque no es «un error grande», es un
colapso: el modelo deja de discriminar.

### `colapso_a_fondo`
El inverso: el modelo no predice casi nada.

Indicador: fracción predicha por debajo de 0,02 cuando la de referencia supera 0,10.

### `fragmentacion`
La región está aproximadamente bien pero troceada en muchas componentes, donde la referencia tiene
una sola.

Indicador: número de componentes conexas de la predicción muy por encima del de la referencia. Las
28 máscaras curadas tienen mediana de 1 componente en Video1 y 14,5 en Video6.

### `borde`
La región es correcta y el contorno no. Es el caso que distingue IoU de F1 de borde.

Indicador: IoU razonable para el video pero F1 de borde claramente por debajo de la media de ese
modelo en ese video.

### `confusion_lecho_marino`
Sedimento, roca o fondo estructurado clasificado como estructura. Error de textura, no de
visibilidad.

### `texto_superpuesto`
Marcas, rótulos o HUD quemados en la imagen que el modelo segmenta como objeto. Afecta
principalmente a Video6.

### `desacuerdo_de_criterio`
**El «error» no es del modelo: es un caso que el criterio 5.0 no resuelve.** Cables, objetos
sueltos, sombras, o una estructura cuyo límite es discutible. La máscara de referencia tomó una
decisión y la predicción tomó la otra, y las dos son defendibles.

Esta categoría es la más importante de mantener separada, por dos razones:

1. No se arregla cambiando el modelo, se arregla cerrando el criterio.
2. Marca el techo real: lo que caiga aquí es lo que la concordancia entre anotadores de la Task
   5.6 debería recoger de forma independiente. Conviene cruzar esta columna con la lista de dudas
   de la segunda anotación (`Registro_Etiquetado/dudas_5_6_*.csv`).

### `otra`
Con nota obligatoria. Si `otra` pasa del 15 % de los casos, la taxonomía está mal y hay que
revisarla antes de seguir clasificando.

---

## Lo que hay que responder con la tabla

El objetivo de la Task 5.13 plantea dos preguntas. Conviene dejar explícito cómo se contestan, para
no terminar con una tabla de frecuencias y ninguna conclusión.

### ¿Los dos modelos fallan en los mismos frames?

Se compara el conjunto de los 10 peores de cada modelo, por video:

```
solape = |peores10(U-Net) ∩ peores10(SegFormer)| / 10
```

- Solape alto significa que la dificultad está en los frames, es decir, en los datos o en las
  etiquetas, y que cambiar de modelo no lo arregla.
- Solape bajo significa que cada arquitectura falla por su cuenta, y entonces la distribución de
  causas de cada una sí es informativa.

### ¿Falla distinto en la tubería real (Video6) que en los cascos?

Se compara la distribución de causas de Video6 contra la del resto de videos, para cada modelo.
Hay motivo previo para esperar que sí: en Video6 la estructura ocupa el 8 % del frame y en Video1
el 72 %, los frames son 480×360 en vez de 640×360, y es el único video con texto superpuesto. Si
las causas dominantes en Video6 son `estructura_pequena` y `sobresegmentacion_global` mientras en
el resto son `borde` y `visibilidad_baja`, la respuesta es que son dos problemas distintos y no
uno con dos niveles de dificultad.

---

## Reglas de la clasificación

- Una causa primaria por caso, obligatoria. Secundaria opcional.
- Ante la duda entre dos, se elige la que **desaparecería primero** si se arreglara: si al mejorar
  la visibilidad el error se iría, es `visibilidad_baja`; si seguiría ahí, es otra.
- La clasificación de cada modelo la hace su responsable (Ariana el U-Net, Dayana el SegFormer),
  pero con esta lista cerrada de antemano y sin modificarla a mitad.
- Si al clasificar aparece una causa que falta, se añade **a este documento** y se revisan los
  casos ya clasificados. No se añade una categoría sobre la marcha en un solo CSV.

---

## Referencias

- `docs/protocolo_evaluacion.md` — protocolo que produce las predicciones.
- `notebooks/2_3_sam_mask_audit.ipynb` — Task 4.3, de donde salen los indicadores de fracción de
  estructura y de fragmentación.
- `Seleccion/seleccion_5_1.csv` — `tipo`, `laplacian_var` y `contraste` por frame.
- `Registro_Etiquetado/dudas_5_6_*.csv` — dudas de la segunda anotación, para cruzar con
  `desacuerdo_de_criterio`.
