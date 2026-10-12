# Competencia 01 — Predicción de BAJA+2 (DMEyF 2026)

Modelo de LightGBM para elegir a qué clientes enviarles un estímulo de retención en **202108**.
La entrega final es un **blend de dos experimentos** (j4955 y j4957) con un corte de **12.000 envíos**.

**Archivo entregado:** `Intentoj_blend_12000.csv`

---

## 1. Estructura del repositorio

```
.
├── README.md
├── notebooks/
│   ├── j4953_competencia.ipynb      # experimento HT4953-01: bayesiana que dio los hiperparámetros del j4955
│   ├── j4955_competencia.ipynb      # experimento HT4955-01 (hiperparámetros del j4953, bynode = 1.0, 3 semillas)
│   ├── j4957_competencia.ipynb      # experimento HT4957-01 (bayesiana propia, bynode = 0.2, 12 semillas)
│   └── blend_j4955_j4957.ipynb      # blend de los dos y archivos de envío
├── bayesianas/
    ├── HT4953-01_BO_log.txt         # log completo de la bayesiana del j4953
    └── HT4957-01_BO_log.txt         # log completo de la bayesiana del j4957

```

Los datasets, los modelos `.txt` de LightGBM y las predicciones intermedias no se suben (ver `.gitignore`).

## 2. Problema y ganancia

- Datos: `competencia_01_crudo.csv`, meses 202103 a 202108.
- `clase_ternaria`: `CONTINUA`, `BAJA+1`, `BAJA+2`. En 202107 y 202108 la clase no se conoce.
- Ganancia por cliente estimulado: **+1.072.500** si es `BAJA+2`, **−27.500** en cualquier otro caso.
- El modelo ordena a los clientes por probabilidad de baja, y la entrega son los primeros N (el **corte**).

## 3. Pipeline (igual en los dos experimentos)

### 3.1 Feature engineering (de 155 a 1.305 columnas)

| Paso | Detalle |
|---|---|
| Datos rotos | `ccajas_depositos` vale 0 en todo 202105, así que se pasa a `NA` en ese mes |
| Flags `_zero` | 140 variables: 1 si la original vale 0, 0 si no, NA si es NA (no se rankean ni se laguean) |
| Totales y ratios de tarjeta | `mv_*` (Master + Visa) y ratios de utilización `mvr_*` (saldo, consumo, pago y pago mínimo sobre el límite), calculados sobre montos crudos |
| Ranking de montos | 82 variables monetarias reemplazadas por su ranking dentro de cada mes (0 queda fijo, positivos en (0,1], negativos en [−1,0)), para eliminar el efecto de la inflación |
| Historia | `lag1`, `lag2`, `delta1`, `delta2` de todas las variables |
| Tendencia | pendiente por cuadrados mínimos (`_tend3`) y promedio (`_avg3`) en una ventana de 3 meses |
| Manuales | `ctrx_quarter_normalizado` (corrige a los clientes con menos de 3 meses de antigüedad) y `mpayroll_sobre_edad` |
| Transiciones | cambios de estado de tarjeta y de mora mes a mes, flags de "prende/apaga" de actividad, payroll y consumos, y `n_canales_apagados` |

### 3.2 Target y pesos

- Clase binaria `clase01` = 1 para `BAJA+1` y `BAJA+2`.
- Peso 0,25 a los `BAJA+1` y 1,0 al resto, para que el modelo priorice los `BAJA+2`.

### 3.3 Limpieza de registros dudosos

Un modelo auxiliar chico (`modelo30`: 2.180 árboles, 30 hojas) puntúa cada registro. Se eliminan los `CONTINUA` con `prob30 > 0.5`, que son clientes que se ven como bajas y confunden al entrenamiento.

- **En la bayesiana:** `modelo30` se entrena con el 30% (fold 2) y limpia el 70% (fold 1). Se eliminan 161 `CONTINUA`.
- **En el modelo final:** *cross-fitting* 50/50 sobre 202103–202106. Cada mitad se puntúa con un modelo entrenado en la otra, así ningún registro se evalúa con un modelo que lo vio. Se eliminan 304 `CONTINUA`.

### 3.4 Optimización bayesiana

- Entrenamiento 202103 + 202104 (fold 1 limpio) y validación 202106. Métrica: AUC.
- `mlr3mbo`, surrogate kriging y expected improvement.
- Espacio de búsqueda: `num_iterations` [64, 4096], `num_leaves` [16, 2048], `min_sum_hessian_in_leaf` [1e-6, 0.1], `feature_fraction` [0.1, 0.9].
- Fijos: `learning_rate` 0,005, `min_data_in_leaf` 0, `max_bin` 31, sin bagging.

### 3.5 Modelo final

- Entrenamiento en **202103 a 202106** (limpio), predicción sobre **202108**.
- Se entrena el mismo modelo con varias semillas y se **promedian las probabilidades**.

## 4. Los dos experimentos

| | **j4955** (HT4955-01) | **j4957** (HT4957-01) |
|---|---|---|
| `feature_fraction_bynode` | 1.0 | 0.2 |
| Hiperparámetros | bayesiana del **j4953** (copiados a mano en el notebook) | bayesiana propia, 40 iteraciones (5,5 h) |
| `num_iterations` | 3606 | 2652 |
| `num_leaves` | 1443 | 1642 |
| `min_sum_hessian_in_leaf` | 0,0875 | 0,0896 |
| `feature_fraction` | 0,120 | 0,110 |
| AUC en la bayesiana | 0,91348 | 0,91353 |
| Semillas en el modelo final | 3 | 12 |

Las dos bayesianas llegaron prácticamente a la misma zona y al mismo AUC. Lo que distingue a los modelos es el `bynode` y la cantidad de semillas.

### 4.1 Origen de los hiperparámetros del j4955

El j4955 **no corre su propia optimización bayesiana**, para no repetir una corrida de varias horas. Usa los mejores hiperparámetros encontrados por la bayesiana del experimento **j4953** (`notebooks/j4953_competencia.ipynb`, log completo en `bayesianas/HT4953-01_BO_log.txt`).

En `j4955_competencia.ipynb`, las celdas de la bayesiana están comentadas y los hiperparámetros se cargan a mano, en la celda bajo el título *"IMPORTANTE, HARDCODEO ESTO PARA NO CORRER TODA LA BAYESIANA"*:

```r
PARAM$out$lgbm$mejores_hiperparametros <- list(
  num_iterations          = 3606,
  num_leaves              = 1443,
  min_sum_hessian_in_leaf = 0.0874518118416699,
  feature_fraction        = 0.120189655432349
)
PARAM$out$lgbm$y <- 0.913479658101473   # AUC de esa configuración en la bayesiana del j4953
```

Esos valores son la **fila de mayor `metrica`** del `BO_log.txt` del j4953.

<!-- COMPLETAR: diferencias de configuración entre j4953 y j4955, si las hay (por ejemplo feature_fraction_bynode o cambios de features). Si no hay ninguna, dejar: "La configuración del j4953 es idéntica a la del j4955; solo cambia que el j4955 no ejecuta la bayesiana." -->

**Cómo proceder para reproducirlo:**

1. Correr `j4953_competencia.ipynb` completo, incluida la bayesiana. Deja `~/buckets/b1/exp/HT4953-01/BO_log.txt`.
2. Tomar la mejor fila de ese log:
   ```r
   tb <- fread("~/buckets/b1/exp/HT4953-01/BO_log.txt")
   setorder(tb, -metrica)
   tb[1, .(num_iterations, num_leaves, min_sum_hessian_in_leaf, feature_fraction, metrica)]
   ```
3. Verificar que coincidan con los valores hardcodeados en el j4955 (o reemplazarlos por los obtenidos) y correr `j4955_competencia.ipynb`.

También se puede omitir el paso 1 y usar directamente los valores del log que está en `bayesianas/HT4953-01_BO_log.txt`.

## 5. Evaluación

### 5.1 Backtest en 202106

Entrenamiento 202103 + 202104 y test en 202106 (que sí tiene clase), con 5 semillas. Ganancia total, en millones:

| Envíos | j4955 (media) | j4957 (media) | j4955 (ensamble) | j4957 (ensamble) |
|---|---|---|---|---|
| 13.200 | 405,5 | 417,3 | 408,1 | 420,2 |
| 14.200 | 400,0 | 416,9 | 403,7 | 426,8 |
| 15.000 | 400,2 | 417,3 | 399,3 | 423,5 |
| Mejor corte | 13.200 → 405,5 | 11.600 → 419,8 | 12.200 → 408,1 | 14.200 → 426,8 |

El j4957 gana en el backtest, pero **ese resultado está sesgado a su favor**: su bayesiana eligió los hiperparámetros validando justamente en 202106. Además, la variación entre semillas del j4957 (~20 M entre la mejor y la peor) es casi del mismo tamaño que la diferencia entre los modelos.

### 5.2 Public leaderboard (202108, 30% de los clientes)

| Envíos | j4955 | j4957 | **Blend** |
|---|---|---|---|
| 10.500 | | | 106,34 |
| 11.000 | 109,84 | 102,25 | 107,17 |
| 11.400 | 109,45 | 103,04 | |
| 11.500 | | | 109,15 |
| 11.600 | 108,16 | 102,60 | |
| 11.800 | 107,00 | 102,19 | |
| 12.000 | 106,70 | 104,20 | **107,77** |
| 12.500 | | | 105,79 |
| 13.000 | | | 104,45 |
| 13.500 | | | 102,19 |
| 14.000 | | | 99,69 |
| 14.200 | | 99,63 | |
| 14.500 | | | 101,92 |

En el public, el j4957 queda por debajo del j4955 en todos los cortes comparables (3 a 8 M), aunque la diferencia se achica hacia 12.000. El blend queda entre los dos modelos con 11.000 envíos, empata con el j4955 alrededor de 11.500 y **con 12.000 supera a ambos** (107,77 contra 106,70 y 104,20). Todas estas diferencias son de 1 a 8 M, o sea 1–7 `BAJA+2`. El public tiene un ruido de alrededor de ±10 M (son ~49 mil clientes y unos 220 `BAJA+2` entre los enviados), así que diferencias de ese tamaño no son concluyentes.

## 6. Blend

### 6.1 Qué hace

Combina los rankings de los dos modelos en uno solo:

1. Cada modelo le asigna una probabilidad a **todos** los clientes de 202108 (`prediccion_promedio.txt`, que ya es el promedio de sus semillas).
2. Cada probabilidad se convierte en un **puesto** dentro de su modelo.
3. Se **promedian los dos puestos** de cada cliente.
4. Se ordena por ese promedio y se envía a los primeros N.

Ejemplo:

| Cliente | Puesto j4955 | Puesto j4957 | Promedio | Puesto final |
|---|---|---|---|---|
| A | 2 | 1 | 1,5 | 1 |
| B | 1 | 3 | 2,0 | 2 |
| C | 3 | 2 | 2,5 | 3 |

Los clientes que **los dos modelos** ponen arriba quedan arriba. Los que solo uno pone arriba bajan a una posición intermedia.

Se promedian **puestos y no probabilidades** para que los dos modelos pesen exactamente la mitad, aunque sus probabilidades estén en escalas distintas. Para elegir los primeros N solo importa el orden.

### 6.2 Por qué un blend

- En el public gana el j4955 y en el backtest gana el j4957 (este último con el sesgo explicado arriba). Las dos señales apuntan para lados opuestos y son del tamaño del ruido. No hay un ganador claro.
- Cada modelo se equivoca en clientes distintos (tienen distinto `bynode` y distintos hiperparámetros). Al promediar, los errores que comete uno solo pierden peso.
- Para el private (el 70% restante) es la opción con menos riesgo: no depende de haber acertado cuál de los dos modelos es mejor.

### 6.3 Elección del corte: 12.000

- En el public, el blend rinde mejor entre 11.000 y 12.000 envíos y baja de forma sostenida a partir de 12.500.
- Con 12.000 envíos, el blend es la mejor de las tres opciones en el public: blend 107,77, j4955 106,70 y j4957 104,20.
- El backtest de 202106 ubicaba el óptimo más arriba (12.200–14.200).
- 12.000 queda en el borde alto de la zona buena del public y más cerca del óptimo del backtest. No se eligió el máximo exacto del public (11.500), porque entre cortes vecinos la diferencia es de 1–2 `BAJA+2`, es decir, ruido.

## 7. Cómo reproducir

Todos los notebooks usan kernel R.

1. *(Opcional)* Correr `notebooks/j4953_competencia.ipynb`, que contiene la bayesiana que dio origen a los hiperparámetros del j4955. Ver la sección 4.1. Si se omite, se usan los valores ya cargados en el j4955, que coinciden con `bayesianas/HT4953-01_BO_log.txt`.
2. Correr `notebooks/j4955_competencia.ipynb`. Genera `~/buckets/b1/exp/HT4955-01/prediccion_promedio.txt`.
3. Correr `notebooks/j4957_competencia.ipynb` (incluye su bayesiana, de ~5,5 h). Genera `~/buckets/b1/exp/HT4957-01/prediccion_promedio.txt`.
4. Correr `notebooks/blend_j4955_j4957.ipynb`. Solo lee esos dos archivos y escribe los CSV de envío en `~/buckets/b1/exp/BLEND-4955-4957/`.

Semilla primigenia: 600013.
