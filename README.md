# SAPCER v2 -- Sistema Agéntico de Predicción de Consumo Energético con Reconocimiento de Contexto

Proyecto de posgrado en Aprendizaje Automático (Machine Learning).
Pipeline multi-agente que lee el dataset histórico energético de Colombia (1975-2024),
entrena y compara cuatro modelos de ML por tipo de energía y sector de consumo,
enriquece las predicciones con un agente de contexto que busca y clasifica noticias
energéticas en tiempo real, y presenta los resultados en un dashboard interactivo Streamlit.

---

## Índice

1. [Motivación y Objetivo](#1-motivación-y-objetivo)
2. [Dataset](#2-dataset)
3. [Arquitectura Multi-Agente](#3-arquitectura-multi-agente)
4. [Pipeline General](#4-pipeline-general)
5. [Agente 1 -- Ingesta y Preprocesamiento](#5-agente-1--ingesta-y-preprocesamiento)
6. [Agente 2 -- Análisis Exploratorio (EDA)](#6-agente-2--análisis-exploratorio-eda)
7. [Agente 3 -- Predictor Multi-Modelo](#7-agente-3--predictor-multi-modelo)
8. [Agente 4 -- Contexto y Noticias](#8-agente-4--contexto-y-noticias)
9. [Agente 5 -- Validador y Comparador](#9-agente-5--validador-y-comparador)
10. [Orquestador](#10-orquestador)
11. [Dashboard Streamlit](#11-dashboard-streamlit)
12. [Estructura del Proyecto](#12-estructura-del-proyecto)
13. [Dependencias e Instalación](#13-dependencias-e-instalación)
14. [Ejecución](#14-ejecución)
15. [Conexión con el Material Teórico del Curso](#15-conexión-con-el-material-teórico-del-curso)
16. [Justificación de Decisiones de Diseño](#16-justificación-de-decisiones-de-diseño)

---

## 1. Motivación y Objetivo

### Problema

La planificación energética de un país requiere anticipar cuánta energía se consumirá
en los próximos años, desglosada por sector económico (industrial, residencial, transporte,
comercial) y por tipo de fuente (gas, petróleo, carbón, hidroenergía, renovables, etc.).
Los modelos puramente estadísticos capturan tendencias históricas, pero ignoran eventos
exógenos como cambios en políticas públicas, crisis de precios del petróleo o avances
en energías renovables que se reflejan primero en la prensa y luego en los datos.

### Objetivo

Construir un sistema multi-agente que:

1. Limpie y estructure el dataset histórico energético de Colombia (1975-2024).
2. Genere análisis exploratorio automático para entender patrones y tendencias.
3. Entrene cuatro modelos de Machine Learning diferentes y los compare rigurosamente.
4. Busque, clasifique y analice noticias energéticas recientes como capa de contexto cualitativo.
5. Presente todo en un dashboard interactivo donde se pueda explorar, comparar y entender.

### Alcance

El sistema predice series temporales anuales para un horizonte configurable (por defecto
5-10 años hacia adelante). No se pretende llegar a 2050; el horizonte se mantiene
razonable para que las predicciones sean defendibles dado el volumen de datos disponible
(50 años de historia).

---

## 2. Dataset

### Archivo fuente

`data/Historico_energetico_completo_COMPLETO.xlsx` -- Hoja 1

### Características generales

| Propiedad         | Valor                                |
|-------------------|--------------------------------------|
| Período           | 1975 - 2024                          |
| Granularidad      | Anual                                |
| Filas             | 50 (una por año)                     |
| Unidad            | Terajoules (TJ)                      |
| Cobertura         | Colombia -- Balance energético nacional |

### Estructura del dataset

El dataset contiene tres bloques de información separados por columnas:

#### Bloque A -- Oferta Interna de Energía Primaria (9 columnas + total)

Son las fuentes de energía en su estado original, antes de ser transformadas.
Representan la producción energética nacional por tipo de recurso.

| Columna           | Descripción                                                     |
|-------------------|-----------------------------------------------------------------|
| `Bagazo`          | Residuo fibroso de la caña de azúcar, usado como biomasa        |
| `Carbón Mineral`  | Carbón extraído de minas, combustible fósil                     |
| `Gas Natural`     | Gas metano extraído de yacimientos                              |
| `Hidroenergía`    | Energía generada por centrales hidroeléctricas                  |
| `Leña`            | Madera usada como combustible directo                           |
| `Petróleo`        | Crudo extraído de pozos petroleros                              |
| `Residuos`        | Residuos orgánicos e industriales usados para generar energía   |
| `Otros Renov.`    | Otras fuentes renovables (solar, eólica, geotérmica, etc.)     |
| `Total Primarios` | Suma de todas las fuentes primarias                             |

#### Bloque B -- Oferta Interna de Energía Secundaria (12 columnas + total)

Son productos energéticos derivados de la transformación de fuentes primarias.
Representan la energía ya procesada y lista para consumo final.

| Columna              | Descripción                                                    |
|----------------------|----------------------------------------------------------------|
| `Alcohol Carb.`      | Alcohol carburante (etanol) mezclado con gasolina              |
| `Biodiesel`          | Combustible renovable derivado de aceites vegetales            |
| `Carbón Leña`        | Carbón vegetal producido a partir de leña                      |
| `Coque`              | Derivado sólido del carbón mineral, usado en siderurgia        |
| `Diesel`             | Combustible derivado del petróleo para transporte y maquinaria |
| `Electricidad SIN`   | Electricidad del Sistema Interconectado Nacional               |
| `Autogen/Cogen`      | Autogeneración y cogeneración de electricidad                  |
| `Fuel Oil`           | Aceite combustible pesado derivado del petróleo                |
| `Gas Industrial`     | Gas procesado para uso industrial                              |
| `GLP`                | Gas Licuado de Petróleo (gas doméstico)                        |
| `Gasolina Motor`     | Gasolina refinada para vehículos                               |
| `Kerosene/Jet`       | Combustible para aviación y uso doméstico                      |
| `Total Secundarios`  | Suma de todas las fuentes secundarias                          |

#### Bloque C -- Consumo Final por Sector (4 columnas + total)

Es la demanda real de energía desglosada por sector económico.
Representa quién consume la energía producida.

| Columna               | Descripción                                                   |
|-----------------------|---------------------------------------------------------------|
| `Industrial`          | Fábricas, manufactura, minería, construcción                  |
| `Residencial`         | Hogares (cocción, calefacción, iluminación, electrodomésticos)|
| `Transporte`          | Vehículos, trenes, aviones, barcos                            |
| `Comercial y Público` | Oficinas, comercios, hospitales, alumbrado público            |
| `Consumo Total`       | Suma de todos los sectores                                    |

### Notas importantes sobre el dataset

1. Todos los valores están en Terajoules (TJ), lo que permite comparar fuentes
   de naturaleza diferente (gas, electricidad, petróleo) en una misma escala.
2. Algunas columnas tienen valores negativos en ciertos años (como Hidroenergía
   o Carbón Mineral). Estos valores negativos representan importaciones netas
   o ajustes contables en el balance energético, donde la producción interna
   fue menor que el consumo y se cubrió con importaciones.
3. Existen dos columnas aparentemente vacías (`Total General.1` y `Total General.2`)
   que se descartan durante la ingesta.
4. Los datos están como tipo `object` en el Excel original debido a la estructura
   de encabezados múltiples; el agente de ingesta se encarga de convertir todo
   a tipos numéricos correctos.

---

## 3. Arquitectura Multi-Agente

### Filosofía de diseño

El proyecto sigue el paradigma de **Agentic IA** presentado en clase: cada agente
es una unidad autónoma con un rol específico, una entrada definida, una salida
definida y una lógica interna propia. Los agentes cooperan secuencialmente para
cumplir la meta global: predecir el consumo energético e informar con contexto.

Cada agente cumple el principio de **percibe, razona y actúa**:
- **Percibe:** recibe datos del agente anterior o de fuentes externas.
- **Razona:** aplica su lógica especializada (limpieza, análisis, modelado, búsqueda).
- **Actúa:** genera una salida estructurada que alimenta al siguiente agente.

### Los 5 agentes + orquestador

| #  | Agente         | Rol                                      | Entrada                  | Salida                           |
|----|----------------|------------------------------------------|--------------------------|----------------------------------|
| 1  | Ingesta        | Limpiar y estructurar el dataset crudo   | Excel crudo              | DataFrame limpio + metadata      |
| 2  | EDA            | Analizar patrones, tendencias, anomalías | DataFrame limpio         | Estadísticas + gráficos + insights |
| 3  | Predictor      | Entrenar 4 modelos y generar predicciones| DataFrame limpio         | Predicciones + métricas por modelo |
| 4  | Contexto       | Buscar y clasificar noticias energéticas | Keywords + APIs          | Noticias clasificadas + sentimiento |
| 5  | Validador      | Comparar modelos y generar ranking final | Predicciones + métricas  | Ranking + visualizaciones comparativas |
| --  | Orquestador    | Coordinar la ejecución secuencial        | Configuración global     | Resultado consolidado de todos los agentes |

---

## 4. Pipeline General

```
+---------------------------------------------------------------------+
|                         ORQUESTADOR                                 |
|                  (coordina todo el flujo)                            |
+----------+----------------------------------------------------------+
           |
           v
+----------------------+
|  Agente 1 * INGESTA  |
|  Excel -> DataFrame   |
|  limpio + metadata   |
+----------+-----------+
           |
           v
+----------------------+
|  Agente 2 * EDA      |
|  Tendencias, ciclos, |
|  correlaciones,      |
|  anomalías           |
+----------+-----------+
           |
           +-----------------------------+
           v                             v
+----------------------+   +--------------------------+
|  Agente 3 * PREDICTOR|   |  Agente 4 * CONTEXTO     |
|  4 modelos ML:       |   |  Búsqueda de noticias,   |
|  - Regresión Lineal  |   |  clasificación por tema,  |
|  - Random Forest     |   |  análisis de sentimiento  |
|  - XGBoost           |   |                           |
|  - MLP               |   |                           |
+----------+-----------+   +--------------+------------+
           |                              |
           +------------------------------+
           v
+--------------------------+
|  Agente 5 * VALIDADOR    |
|  Compara modelos,        |
|  genera ranking,         |
|  métricas finales        |
+----------+---------------+
           |
           v
+--------------------------+
|  DASHBOARD STREAMLIT     |
|  Visualización           |
|  interactiva de todo     |
+--------------------------+
```

El Agente 3 (Predictor) y el Agente 4 (Contexto) se ejecutan en paralelo conceptualmente:
el predictor trabaja con los datos históricos mientras el agente de contexto busca noticias
externas. Ambos resultados convergen en el Agente 5 (Validador) que los presenta juntos.

---

## 5. Agente 1 -- Ingesta y Preprocesamiento

**Archivo:** `agentes/ingesta.py`

### Responsabilidad

Transformar el archivo Excel crudo, que tiene encabezados MultiIndex y tipos mixtos,
en un DataFrame limpio, tipado y listo para análisis. Este agente es la puerta de entrada
de todo el sistema; si los datos no están limpios, nada posterior funciona.

### Flujo interno

1. **Lectura del Excel:**
   Lee la primera hoja del archivo con `header=[0,1]` para capturar los dos niveles
   de encabezado (grupo de energía + tipo específico). El MultiIndex resultante se
   aplana en nombres de columna legibles.

2. **Renombrado de columnas:**
   Transforma los nombres MultiIndex en nombres cortos y claros:
   - `('Oferta Interna para Consumo Final - PRIMARIOS (TJ)', 'Gas Natural')` -> `primario_gas_natural`
   - `('Consumo Final por Sector (TJ)', 'Industrial')` -> `consumo_industrial`
   - `('Unnamed: 0_level_0', 'Año')` -> `anio`

3. **Eliminación de columnas vacías:**
   Descarta `Total General.1` y `Total General.2` que están completamente vacías (NaN).

4. **Conversión de tipos:**
   Convierte todas las columnas numéricas de `object` a `float64`. Los valores que
   no se pueden convertir (como celdas con texto residual) se convierten a NaN.

5. **Validación de integridad:**
   - Verifica que `anio` sea un rango continuo sin huecos (1975-2024).
   - Reporta el porcentaje de NaN por columna.
   - Verifica que los totales sean consistentes con la suma de sus componentes
     (con una tolerancia del 1% para errores de redondeo).

6. **Generación de metadata:**
   Produce un diccionario con información del dataset:
   - Rango temporal, cantidad de filas y columnas.
   - Lista de series de producción primaria, secundaria y consumo por sector.
   - Estadísticas básicas (mín, máx, media) por columna.

### Salida

```python
{
    "df_clean": pd.DataFrame,          # DataFrame limpio con columnas renombradas
    "metadata": {
        "anio_inicio": 1975,
        "anio_fin": 2024,
        "n_filas": 50,
        "series_primarias": ["bagazo", "carbon_mineral", ...],
        "series_secundarias": ["alcohol_carb", "biodiesel", ...],
        "series_consumo": ["industrial", "residencial", ...],
        "columnas_con_nulos": {...},
        "estadisticas": {...}
    }
}
```

---

## 6. Agente 2 -- Análisis Exploratorio (EDA)

**Archivo:** `agentes/eda.py`

### Responsabilidad

Analizar los datos limpios para descubrir patrones, tendencias, estacionalidad,
correlaciones y anomalías antes de entrenar cualquier modelo. Este agente genera
los insights que informan las decisiones del predictor y provee las visualizaciones
de la sección de datos históricos del dashboard.

### Análisis realizados

#### 6.1 Análisis de tendencias

Para cada serie temporal, calcula:
- **Tendencia lineal:** pendiente de regresión lineal simple sobre el tiempo.
  Una pendiente positiva indica crecimiento sostenido; negativa, declive.
- **Tasa de crecimiento anual compuesta (CAGR):**
  `CAGR = (valor_final / valor_inicial)^(1/n_años) - 1`
  Expresa el crecimiento como porcentaje anual equivalente.
- **Cambios estructurales:** detecta los años donde la serie experimenta un cambio
  abrupto de nivel o pendiente, usando diferencias porcentuales año a año.
  Un cambio mayor al 15% se marca como potencial punto de quiebre.

#### 6.2 Análisis de correlaciones

Calcula la matriz de correlación de Pearson entre todas las series para identificar:
- **Correlaciones fuertes positivas (> 0.8):** series que se mueven juntas
  (ej: consumo industrial y diesel probablemente correlacionan alto).
- **Correlaciones negativas (< -0.5):** series inversamente relacionadas
  (ej: leña decreciendo mientras electricidad crece).
- **Correlaciones con el año:** indica qué series tienen tendencia temporal fuerte.

#### 6.3 Análisis de composición

Calcula cómo ha evolucionado la participación porcentual de cada fuente en el total:
- Participación de cada energía primaria en el total primario.
- Participación de cada energía secundaria en el total secundario.
- Participación de cada sector en el consumo total.

Esto permite ver la transición energética: por ejemplo, si las renovables ganan
participación mientras el carbón la pierde.

#### 6.4 Detección de anomalías

Identifica años atípicos usando el criterio del rango intercuartílico (IQR):
- Calcula Q1, Q3 e IQR para cada serie.
- Marca como anomalía cualquier valor fuera de [Q1 - 1.5xIQR, Q3 + 1.5xIQR].
- Reporta el año y el valor anómalo para contextualización (ej: crisis de 1999,
  pandemia de 2020).

### Salida

```python
{
    "tendencias": {
        "consumo_industrial": {"pendiente": 3200.5, "cagr": 0.023, "tendencia": "creciente"},
        ...
    },
    "correlaciones": pd.DataFrame,     # Matriz de correlación completa
    "composicion": pd.DataFrame,       # % de participación por año
    "anomalias": {
        "consumo_transporte": [{"anio": 2020, "valor": 180000, "tipo": "bajo"}],
        ...
    },
    "graficos": {                      # Figuras Plotly para el dashboard
        "series_historicas": fig1,
        "matriz_correlacion": fig2,
        "composicion_apilada": fig3,
        "anomalias_marcadas": fig4
    }
}
```

---

## 7. Agente 3 -- Predictor Multi-Modelo

**Archivo:** `agentes/predictor.py`

### Responsabilidad

Entrenar cuatro modelos de Machine Learning sobre cada serie temporal del dataset,
generar predicciones hacia el futuro y calcular métricas de rendimiento para cada modelo.
Este es el agente central del sistema y el que concentra la complejidad de ML.

### Estrategia de modelado

#### El problema fundamental: datos anuales con 50 observaciones

Con solo 50 puntos de datos anuales no se puede hacer un split temporal tradicional
(70/15/15) y esperar que los modelos generalicen bien. La estrategia adoptada es:

1. **Feature engineering temporal:** en lugar de alimentar solo el valor de la serie,
   se construyen features derivadas del tiempo que enriquecen cada observación.
2. **Validación con TimeSeriesSplit:** se usa validación cruzada específica para
   series temporales (nunca se usa el futuro para predecir el pasado).
3. **Regularización fuerte:** todos los modelos usan regularización para evitar
   sobreajuste dado el tamaño pequeño del dataset.

#### Feature engineering

Para cada serie temporal, se generan las siguientes features a partir de la
columna `anio` y los valores históricos:

| Feature            | Fórmula / Descripción                              | Propósito                        |
|--------------------|-----------------------------------------------------|----------------------------------|
| `anio`             | Año tal cual (1975, 1976, ...)                      | Captura tendencia lineal         |
| `anio_norm`        | `(anio - anio_min) / (anio_max - anio_min)`         | Normalizado a [0,1] para la MLP  |
| `anio_cuadrado`    | `anio_norm^2`                                        | Captura tendencias no lineales   |
| `lag_1`            | Valor del año anterior                              | Autocorrelación de orden 1       |
| `lag_2`            | Valor de hace 2 años                                | Autocorrelación de orden 2       |
| `lag_3`            | Valor de hace 3 años                                | Autocorrelación de orden 3       |
| `media_movil_3`    | Promedio de los últimos 3 años                      | Suaviza ruido, captura tendencia |
| `media_movil_5`    | Promedio de los últimos 5 años                      | Tendencia de mediano plazo       |
| `diff_1`           | `valor(t) - valor(t-1)`                             | Tasa de cambio absoluto          |
| `diff_pct_1`       | `(valor(t) - valor(t-1)) / valor(t-1)`              | Tasa de cambio porcentual        |

Al crear lags y medias móviles se pierden las primeras filas (hasta 5 según el lag
más largo), quedando ~45 observaciones efectivas para entrenamiento.

### Los 4 modelos

#### 7.1 Regresión Lineal con Ridge (Baseline)

**Implementación:** `sklearn.linear_model.Ridge`

**Justificación:** El notebook de taxonomía del curso recomienda siempre empezar
con un modelo lineal como baseline. Ridge añade regularización L2 que penaliza
coeficientes grandes, previniendo sobreajuste con pocas muestras.

**Ecuación:**

```
y_pred = w_1*anio_norm + w_2*anio^2 + w_3*lag_1 + w_4*lag_2 + ... + b

Pérdida = MSE + alpha x SUM(w_i^2)
```

**Hiperparámetros:**
- `alpha`: fuerza de regularización L2. Se busca con cross-validation entre
  [0.01, 0.1, 1.0, 10.0, 100.0].

**Fortalezas para este problema:**
- Rápido de entrenar, completamente interpretable.
- Los coeficientes indican qué features son más importantes.
- Estable con pocos datos gracias a la regularización.

**Debilidades:**
- Solo captura relaciones lineales entre features y target.
- Si la tendencia energética tiene forma no lineal (ej: crecimiento logístico),
  lo aproximará pobremente.

**Preprocesamiento requerido:** StandardScaler (los modelos lineales son sensibles
a la escala de las features, como se señala en la sección 2 del notebook de taxonomía).

---

#### 7.2 Random Forest Regressor

**Implementación:** `sklearn.ensemble.RandomForestRegressor`

**Justificación:** En la sección 5 del notebook de taxonomía, Random Forest destaca
por su robustez y capacidad de feature importance. El cheat sheet del notebook lo
recomienda como "segundo intento" después del baseline lineal.

**Cómo funciona en nuestro contexto:**

Random Forest entrena múltiples árboles de decisión sobre subconjuntos aleatorios
de los datos (bagging) y promedia sus predicciones. Cada árbol hace splits del tipo
"si lag_1 > 500000 y media_movil_3 > 480000 -> predice 520000".

```
Predicción = promedio(Árbol_1(x), Árbol_2(x), ..., ÁrbolB(x))
```

**Hiperparámetros:**
- `n_estimators`: 100-300 árboles (más árboles = más estable, no overfittea).
- `max_depth`: 3-6 (limitado por la cantidad de datos; profundidad > 6 con 45
  muestras lleva a sobreajuste severo).
- `min_samples_leaf`: 3-5 (cada hoja debe tener al menos 3 observaciones).

**Fortalezas para este problema:**
- No necesita estandarización (invariante a escala, como dice el notebook).
- Captura relaciones no lineales.
- Feature importance muestra qué variables temporales son más predictivas.
- Resistente a outliers.

**Debilidades:**
- No puede extrapolar más allá del rango observado en entrenamiento; si la serie
  crece monótonamente, RF tiende a "aplanarse" en la predicción futura.
- Menos interpretable que la regresión lineal (caja negra parcial).

---

#### 7.3 XGBoost Regressor

**Implementación:** `xgboost.XGBRegressor`

**Justificación:** En la sección 6 del notebook de taxonomía, XGBoost se presenta
como "el estado del arte en datos tabulares" y "Kaggle king". A diferencia de
Random Forest que entrena árboles en paralelo, XGBoost los entrena secuencialmente,
donde cada nuevo árbol corrige los errores (residuos) del anterior.

**Cómo funciona en nuestro contexto:**

```
Paso 1: Modelo base predice la media de la serie
Paso 2: Árbol 1 se entrena sobre los residuos del paso 1
Paso 3: Árbol 2 se entrena sobre los residuos del paso 2
...
Prediccion final = modelo_base + lr*Arbol_1 + lr*Arbol_2 + ... + lr*Arbol_n
```

La tasa de aprendizaje (`lr`) controla cuánto aporta cada nuevo árbol, previniendo
sobreajuste. Esto conecta directamente con la presentación de descenso del gradiente
del curso: cada paso es literalmente un paso de gradient descent en el espacio funcional.

**Hiperparámetros:**
- `n_estimators`: 100-500.
- `max_depth`: 2-4 (árboles poco profundos para evitar overfitting).
- `learning_rate`: 0.01-0.1 (tasa de aprendizaje baja = más robusto).
- `reg_alpha` (L1) y `reg_lambda` (L2): regularización sobre los pesos de las hojas.
- `subsample`: 0.7-0.9 (usa un subconjunto de datos por árbol, como bagging).

**Fortalezas para este problema:**
- Maneja bien relaciones complejas con features temporales.
- La regularización integrada (L1 + L2) previene sobreajuste.
- Generalmente supera a Random Forest en datos tabulares.

**Debilidades:**
- Al igual que Random Forest, tiene dificultad para extrapolar.
- Más hiperparámetros que ajustar.
- Ligeramente más opaco que la regresión lineal.

---

#### 7.4 MLP Regressor (Red Neuronal)

**Implementación:** `sklearn.neural_network.MLPRegressor`

**Justificación:** En la sección 8 del notebook de taxonomía, el MLP se presenta
como la red neuronal feedforward clásica que se entrena con backpropagation. Este
modelo conecta directamente con la presentación de descenso del gradiente del curso,
que dedicó varias secciones al algoritmo de Adam, el optimizador usado por el MLP.

**Cómo funciona en nuestro contexto:**

```
Capa de entrada:  10 features temporales
         |
Capa oculta 1:    32 neuronas + ReLU
         |
Capa oculta 2:    16 neuronas + ReLU
         |
Capa de salida:   1 neurona (predicción del valor)
```

La arquitectura se mantiene deliberadamente pequeña (32-16) porque con solo 45
muestras de entrenamiento, una red más grande memorizaría los datos en lugar de
aprender patrones generalizables.

**Ecuaciones (del notebook y la presentación):**

```
h_1 = ReLU(W_1 * x + b_1)       # Capa oculta 1
h_2 = ReLU(W_2 * h_1 + b_2)      # Capa oculta 2
y_pred  = W_3 * h_2 + b_3             # Capa de salida

Pérdida = MSE(y, y_pred)

Actualización de pesos con Adam:
  m_t = beta_1*m_t-_1 + (1-beta_1)*g_t          (momento de primer orden)
  v_t = beta_2*v_t-_1 + (1-beta_2)*g_t^2         (momento de segundo orden)
  theta_t+_1 = theta_t - eta/sqrt(v_t+epsilon) * m_t        (actualización adaptativa)
```

**Hiperparámetros:**
- `hidden_layer_sizes`: (32, 16) -- arquitectura de 2 capas.
- `solver`: 'adam' -- el optimizador estudiado en clase.
- `learning_rate_init`: 0.001 -- tasa de aprendizaje inicial.
- `max_iter`: 2000 -- épocas máximas de entrenamiento.
- `early_stopping`: True -- detiene si la validación no mejora.
- `alpha`: 0.01 -- regularización L2 sobre los pesos.

**Preprocesamiento requerido:** StandardScaler obligatorio (la presentación de
descenso del gradiente explica cómo las escalas dispares hacen que el optimizador
oscile o diverja).

**Fortalezas para este problema:**
- Puede capturar relaciones no lineales complejas.
- Valor pedagógico directo: aplica descenso del gradiente y Adam en la práctica.

**Debilidades:**
- Con 45 muestras, el riesgo de sobreajuste es alto incluso con regularización.
- Caja negra completa (no hay interpretabilidad directa).
- Sensible a la inicialización de pesos y a los hiperparámetros.

---

### Estrategia de validación temporal

No se usa validación aleatoria (que mezclaría pasado y futuro). Se usa
`TimeSeriesSplit` de sklearn con 5 splits:

```
Split 1: Train [1975-1995] -> Test [1996-2000]
Split 2: Train [1975-2000] -> Test [2001-2005]
Split 3: Train [1975-2005] -> Test [2006-2010]
Split 4: Train [1975-2010] -> Test [2011-2015]
Split 5: Train [1975-2015] -> Test [2016-2020]

Evaluación final: Train [1975-2020] -> Test [2021-2024]
```

Esto simula el escenario real: entrenar con historia y predecir el futuro.

### Métricas calculadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| RMSE    | `sqrt(SUM(y_i-y_pred_i)^2 / n)` | Error promedio en las mismas unidades (TJ) |
| MAE     | `SUM|y_i-y_pred_i| / n` | Error absoluto promedio, menos sensible a outliers |
| MAPE    | `SUM|y_i-y_pred_i|/|y_i| x 100 / n` | Error porcentual, comparable entre series |
| R^2      | `1 - SUM(y_i-y_pred_i)^2 / SUM(y_i-y_mean)^2` | Proporción de varianza explicada (1=perfecto) |

### Salida del agente

```python
{
    "predicciones": {
        "consumo_industrial": {
            "ridge": {"historico": [...], "futuro": [...], "metricas": {...}},
            "random_forest": {"historico": [...], "futuro": [...], "metricas": {...}},
            "xgboost": {"historico": [...], "futuro": [...], "metricas": {...}},
            "mlp": {"historico": [...], "futuro": [...], "metricas": {...}},
        },
        ...
    },
    "ranking_global": pd.DataFrame,     # Ranking de modelos por métrica
    "feature_importance": {              # Solo RF y XGBoost
        "random_forest": pd.DataFrame,
        "xgboost": pd.DataFrame,
    },
    "mejor_modelo_por_serie": {...}      # Qué modelo ganó en cada serie
}
```

---

## 8. Agente 4 -- Contexto y Noticias

**Archivo:** `agentes/contexto.py`

### Responsabilidad

Buscar noticias recientes sobre el sector energético, clasificarlas por temática,
analizar su sentimiento y presentarlas como una capa de contexto cualitativo que
acompaña las predicciones numéricas del Agente Predictor.

Este agente NO modifica las predicciones matemáticas. Su rol es informativo:
provee al usuario la información contextual que un modelo numérico puro no puede
capturar. Cuando el dashboard muestra "consumo de gas crecerá 3% según XGBoost",
el agente de contexto puede mostrar al lado "se detectaron 7 noticias sobre
nueva regulación de gas natural que podrían afectar esta tendencia".

### Flujo interno

#### 8.1 Búsqueda de noticias

El agente busca noticias usando web search con queries predefinidos y configurables:

```python
QUERIES_BUSQUEDA = [
    "consumo energético Colombia 2024 2025",
    "precio petróleo impacto Colombia",
    "transición energética Colombia",
    "crisis energética Latinoamérica",
    "energía renovable Colombia inversión",
    "gas natural Colombia regulación",
    "fenómeno del Niño energía Colombia",
]
```

Busca en fuentes de noticias generales. No requiere API keys de pago;
usa búsqueda web pública y extracción de contenido.

#### 8.2 Clasificación por categorías

Cada noticia se clasifica en una de cinco categorías temáticas mediante
análisis de keywords en el título y cuerpo del texto:

| Categoría             | Keywords asociados                                            | Ícono |
|-----------------------|---------------------------------------------------------------|-------|
| Politicas energeticas | regulacion, ley, decreto, ministerio, CREG, subsidio, reforma| POL  |
| Crisis / Escasez      | crisis, apagon, racionamiento, escasez, emergencia, sequia   | CRI  |
| Economia y Precios    | precio, inversion, PIB, exportacion, OPEP, dolar, inflacion  | ECO  |
| Transicion Verde      | renovable, solar, eolica, hidrogeno, descarbonizacion, COP   | VER  |
| Infraestructura       | planta, proyecto, construccion, linea, transmision, expansion | INF  |

Si una noticia contiene keywords de múltiples categorías, se asigna a la que
tenga más coincidencias. Si hay empate, se clasifica como la primera en la lista
de prioridades (las crisis tienen prioridad sobre lo demás).

#### 8.3 Análisis de sentimiento

Para cada noticia se calcula un score de sentimiento simple basado en la
proporción de palabras positivas vs negativas en el título y resumen:

```
sentimiento = (n_palabras_positivas - n_palabras_negativas) / n_palabras_totales
```

- **Positivo (> 0.1):** crecimiento, inversión, mejora, récord, avance, expansión
- **Negativo (< -0.1):** crisis, caída, riesgo, recorte, problema, déficit, apagón
- **Neutro (-0.1 a 0.1):** informativo, estadístico, descriptivo

#### 8.4 Generación del resumen contextual

El agente genera un resumen estructurado:

```python
{
    "noticias": [
        {
            "titulo": "Colombia acelera plan de transición energética",
            "fuente": "Reuters",
            "fecha": "2025-03-15",
            "categoria": "Transición Verde",
            "sentimiento": 0.35,
            "sentimiento_label": "positivo",
            "resumen": "...",
            "url": "https://..."
        },
        ...
    ],
    "resumen": {
        "total_noticias": 15,
        "distribucion_categorias": {
            "Políticas energéticas": 4,
            "Crisis / Escasez": 2,
            "Economía y Precios": 5,
            "Transición Verde": 3,
            "Infraestructura": 1
        },
        "sentimiento_promedio": 0.12,
        "sentimiento_label": "ligeramente positivo",
        "alertas": [
            "[ALERTA] Se detectaron 2 noticias de crisis energética recientes"
        ]
    }
}
```

### ¿Por qué el agente de contexto no ajusta las predicciones?

Decisión de diseño deliberada por tres razones:

1. **Rigor metodológico:** Un modelo de ML se valida con métricas cuantitativas.
   Si se le aplica un factor cualitativo basado en noticias, se contamina la
   evaluación y no se puede saber si la predicción mejoró por el modelo o por suerte
   en la interpretación de noticias.

2. **Separación de responsabilidades:** La predicción numérica y la contextualización
   informativa son dos tareas diferentes. Mezclarlas haría más difícil diagnosticar
   problemas y mejorar cada componente por separado.

3. **Honestidad epistémica:** Es más valioso mostrar al usuario "el modelo predice X
   y el contexto actual sugiere Y" que dar un solo número que mezcla ambas cosas
   de forma opaca. El usuario puede tomar una decisión informada con ambas fuentes.

---

## 9. Agente 5 -- Validador y Comparador

**Archivo:** `agentes/validador.py`

### Responsabilidad

Recibir los resultados de todos los modelos del Agente Predictor, compararlos
rigurosamente y generar un ranking final con visualizaciones que permitan al
usuario entender no solo cuál modelo es mejor, sino por qué y en qué condiciones.

### Análisis que realiza

#### 9.1 Tabla comparativa de métricas

Para cada serie temporal y cada modelo, genera una tabla con todas las métricas:

```
Serie: consumo_industrial
+----------------+---------+---------+---------+-------+
| Modelo         | RMSE    | MAE     | MAPE(%) | R^2    |
+----------------+---------+---------+---------+-------+
| Ridge          | 15,200  | 12,100  | 4.2     | 0.91  |
| Random Forest  | 13,800  | 11,500  | 3.8     | 0.93  |
| XGBoost        | 12,100  | 10,200  | 3.3     | 0.95  |
| MLP            | 16,500  | 13,800  | 4.8     | 0.89  |
+----------------+---------+---------+---------+-------+
```

#### 9.2 Ranking global

Promedia las métricas de cada modelo a través de todas las series temporales
y genera un ranking. Se usa MAPE como métrica principal porque es comparable
entre series de diferentes magnitudes.

#### 9.3 Análisis de residuos

Para cada modelo, calcula y visualiza:
- **Distribución de residuos:** ¿son normales y centrados en cero? Si no,
  el modelo tiene un sesgo sistemático.
- **Residuos vs tiempo:** ¿los errores son más grandes en ciertos períodos?
  Si crecen al final, el modelo no captura cambios recientes.
- **Autocorrelación de residuos:** ¿los errores están correlacionados?
  Si sí, el modelo no está capturando toda la estructura temporal.

#### 9.4 Visualización de predicciones vs realidad

Gráficos superpuestos donde se muestra:
- Serie histórica real (línea negra sólida).
- Predicción in-sample de cada modelo (líneas de colores punteadas).
- Predicción futura de cada modelo (líneas de colores sólidas con anotación).
- Zona de predicción futura sombreada.

#### 9.5 Análisis de feature importance comparativo

Compara qué features son más importantes para Random Forest vs XGBoost.
Si ambos coinciden en que `lag_1` y `media_movil_5` son las más importantes,
hay consenso en que la inercia reciente es lo que más predice. Si difieren
significativamente, cada modelo está capturando patrones diferentes.

### Salida

```python
{
    "ranking_global": pd.DataFrame,
    "metricas_por_serie": dict,
    "mejor_modelo_por_serie": dict,
    "analisis_residuos": dict,
    "graficos_comparativos": dict,       # Figuras Plotly
    "resumen_textual": str               # Texto explicativo para el dashboard
}
```

---

## 10. Orquestador

**Archivo:** `orquestador.py`

### Responsabilidad

Coordinar la ejecución secuencial de los 5 agentes, manejar el estado global
del pipeline, controlar errores y generar logs de ejecución.

### Flujo de ejecución

```python
orquestador.ejecutar(
    ruta_dataset    = "data/Historico_energetico_completo_COMPLETO.xlsx",
    horizonte       = 10,                # Años a predecir hacia el futuro
    series_objetivo = "todas",           # O lista específica de series
    buscar_noticias = True,              # Activar/desactivar agente de contexto
    verbose         = True,              # Imprimir progreso en consola
)
```

### Gestión del estado

El orquestador mantiene un diccionario de estado que se pasa entre agentes:

```python
estado = {
    "paso_actual": "ingesta",
    "inicio": datetime,
    "config": {...},
    "resultados": {
        "ingesta": None,       # Se llena cuando termina el agente 1
        "eda": None,           # Se llena cuando termina el agente 2
        "predictor": None,     # Se llena cuando termina el agente 3
        "contexto": None,      # Se llena cuando termina el agente 4
        "validador": None,     # Se llena cuando termina el agente 5
    },
    "errores": [],
    "duracion_total_seg": None
}
```

### Manejo de errores

Si un agente falla, el orquestador:
1. Registra el error con traceback completo.
2. Intenta continuar con los agentes restantes si es posible (ej: si el
   agente de contexto falla por falta de internet, el predictor no se afecta).
3. Reporta al dashboard qué agentes completaron exitosamente y cuáles no.

---

## 11. Dashboard Streamlit

**Archivo:** `dashboard.py`

### Estructura visual

El dashboard tiene 5 pestañas principales:

#### Pestaña 1 -- Inicio y Resumen Ejecutivo

- Métricas KPI en tarjetas: consumo total más reciente, crecimiento anual,
  mejor modelo global, número de noticias analizadas.
- Mini gráfico de la serie de consumo total con la predicción ganadora.

#### Pestaña 2 -- Datos Históricos

- Selector interactivo de serie temporal (dropdown con las ~25 series).
- Gráfico de línea interactivo (Plotly) con zoom, hover, descarga.
- Gráfico de composición apilada (evolución de la mezcla energética).
- Matriz de correlación como heatmap interactivo.
- Tabla de anomalías detectadas.

#### Pestaña 3 -- Modelos y Predicciones

- Selector de serie temporal.
- Gráfico principal: serie histórica + predicciones de los 4 modelos superpuestas.
- Tabla comparativa de métricas (RMSE, MAE, MAPE, R^2) con el mejor resaltado.
- Gráfico de barras comparativo de métricas.
- Feature importance (para RF y XGBoost).
- Gráfico de residuos del modelo seleccionado.

#### Pestaña 4 -- Contexto Energético

- Timeline de noticias recientes con íconos por categoría.
- Gráfico de torta con distribución por categoría.
- Indicador de sentimiento promedio (tipo gauge).
- Lista expandible de noticias con título, fuente, fecha, sentimiento y enlace.
- Alertas si hay noticias de crisis.

#### Pestaña 5 -- Comparativa Final

- Ranking global de modelos (tabla con medallas).
- Gráfico de radar comparativo (cada eje es una métrica).
- Mapa de calor: modelo vs serie temporal (qué modelo gana en cada serie).
- Resumen textual generado automáticamente.

---

## 12. Estructura del Proyecto

```
proyecto_energia/
|
+-- dashboard.py              # Punto de entrada visual (Streamlit)
+-- orquestador.py            # Coordinador del pipeline de agentes
+-- config.py                 # Configuración global (rutas, hiperparámetros, queries)
+-- requirements.txt          # Dependencias del proyecto
+-- README.md                 # Este archivo
|
+-- agentes/
|   +-- __init__.py
|   +-- ingesta.py            # Agente 1 -- Limpieza y estructuración del dataset
|   +-- eda.py                # Agente 2 -- Análisis exploratorio automático
|   +-- predictor.py          # Agente 3 -- Entrenamiento de 4 modelos ML
|   +-- contexto.py           # Agente 4 -- Búsqueda y clasificación de noticias
|   +-- validador.py          # Agente 5 -- Comparación de modelos y ranking
|
+-- data/
|   +-- Historico_energetico_completo_COMPLETO.xlsx   # Dataset fuente
|
+-- outputs/                  # Generado automáticamente al ejecutar
    +-- data_clean.csv        # Dataset procesado por el agente de ingesta
    +-- predicciones/         # CSVs con predicciones por modelo y serie
    +-- noticias/             # JSONs con noticias recopiladas y clasificadas
    +-- figuras/              # Gráficos exportados (opcional)
```

---

## 13. Dependencias e Instalación

### Requisitos del sistema

- Python 3.9 o superior

### Dependencias

```
# Datos y procesamiento
pandas>=2.0
numpy>=1.24
openpyxl>=3.1               # Lectura de archivos Excel

# Machine Learning
scikit-learn>=1.3
xgboost>=2.0

# Visualización
plotly>=5.18
streamlit>=1.30

# Búsqueda de noticias (Agente de Contexto)
requests>=2.31
beautifulsoup4>=4.12        # Parsing de HTML para extracción de noticias
```

### Instalación

```bash
cd proyecto_energia
pip install -r requirements.txt
```

---

## 14. Ejecución

### Dashboard completo (modo interactivo)

```bash
cd proyecto_energia
streamlit run dashboard.py
```

Se abre en el navegador en `http://localhost:8501`.

### Pipeline por consola (modo script)

```python
from orquestador import Orquestador

orq = Orquestador()
resultados = orq.ejecutar(
    horizonte=10,
    buscar_noticias=True,
    verbose=True
)

print(resultados["metadata"]["duracion_total_seg"])
```

---

## 15. Conexión con el Material Teórico del Curso

### Presentación: Descenso del Gradiente y sus Variantes

| Concepto de la presentación                   | Dónde se aplica en el proyecto                                  |
|-----------------------------------------------|----------------------------------------------------------------|
| Ecuación fundamental theta_t+_1 = theta_t - etanabla_L(theta_t)     | Entrenamiento del MLP Regressor (backpropagation)               |
| Adam (Adaptive Moment Estimation)             | Optimizador del MLP (solver='adam')                             |
| Tasa de aprendizaje y su efecto               | Hiperparámetro `learning_rate_init` del MLP                     |
| Gradient Boosting como descenso funcional     | XGBoost: cada árbol es un paso de gradient descent sobre residuos|
| SGD Classifier                                | SGD es la base del entrenamiento de Ridge con grandes datasets   |
| Sensibilidad a inicialización                 | `random_state` fijo para reproducibilidad en MLP y RF            |

### Notebook: Taxonomía de Modelos ML

| Sección del notebook              | Modelo en el proyecto                                   |
|-----------------------------------|---------------------------------------------------------|
| Sección 2: Modelos Lineales      | Ridge Regressor (baseline)                               |
| Sección 5: Basados en Árboles    | Random Forest Regressor                                  |
| Sección 6: Boosting              | XGBoost Regressor                                        |
| Sección 8: Redes Neuronales      | MLP Regressor                                            |
| Sección 10: Gran Comparación     | Agente Validador (ranking de modelos con múltiples métricas)|
| Sección 12: Impacto de PCA       | Feature engineering como alternativa a PCA para series temporales|
| Cheat Sheet                       | Decisión de usar baseline + RF + XGBoost + MLP            |

### Presentación: Proyectos Agentic IA

| Concepto del proyecto de clase             | Equivalente en SAPCER v2                                 |
|--------------------------------------------|----------------------------------------------------------|
| Agente de percepción (CNN clasifica)       | Agente Predictor (modelos ML predicen)                   |
| Agente de conocimiento (info.json)         | Agente de Contexto (noticias como fuente de conocimiento)|
| Agente de interacción (Tkinter/Kivy)       | Dashboard Streamlit                                      |
| Orquestador de flujo                       | Orquestador.py                                           |
| Cada agente percibe, razona y actúa        | Cada agente tiene entrada, lógica interna y salida       |
| Ecosistema multi-agente cooperativo        | 5 agentes + orquestador trabajando en secuencia          |

---

## 16. Justificación de Decisiones de Diseño

### ¿Por qué NO se usa LSTM?

El notebook de taxonomía del curso no incluye LSTM (pertenece a PyTorch puro,
fuera de sklearn). Con 50 puntos de datos anuales, un LSTM típico tiene más
parámetros entrenables que observaciones disponibles, lo que lleva a sobreajuste
severo. La presentación de descenso del gradiente advierte sobre el vanishing
gradient en redes profundas, problema que se agrava con secuencias cortas.
El MLP del notebook cumple el mismo rol pedagógico de "red neuronal" con
más solidez estadística para este volumen de datos.

### ¿Por qué el agente de contexto no modifica las predicciones?

Ver sección 8 para la justificación completa. En resumen: rigor metodológico
(las métricas del modelo deben ser puras), separación de responsabilidades
(predicción numérica != interpretación cualitativa) y honestidad epistémica
(mejor dar dos fuentes de información que una mezcla opaca).

### ¿Por qué TimeSeriesSplit y no validación aleatoria?

En series temporales, el orden importa. Usar validación aleatoria (como
KFold) permitiría que datos futuros informen la predicción del pasado,
inflando artificialmente las métricas. TimeSeriesSplit simula el uso real:
siempre se entrena con el pasado y se evalúa con el futuro.

### ¿Por qué features temporales y no el valor crudo?

Un modelo de regresión necesita features como entrada. Si se le da solo el
año como feature, solo puede aprender una relación año->valor (regresión simple).
Al agregar lags, medias móviles y diferencias, el modelo puede aprender patrones
como "si los últimos 3 años crecieron en promedio X%, el próximo año crecerá
aproximadamente Y%". Esto transforma el problema de forecasting en un problema
de regresión tabular estándar, donde los modelos del notebook (RF, XGBoost,
MLP) son directamente aplicables.

### ¿Por qué 4 modelos y no 1?

El notebook de taxonomía dedica 12 secciones a demostrar que no existe un modelo
universalmente mejor. Diferentes familias capturan diferentes patrones:
- Ridge captura tendencias lineales.
- RF captura no-linealidades sin extrapolar.
- XGBoost corrige errores iterativamente.
- MLP aproxima funciones arbitrarias.

Comparar los cuatro es tanto una práctica de ML rigurosa como un requisito
pedagógico del curso.
