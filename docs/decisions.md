# Decisiones arquitectónicas

Este documento registra las decisiones de diseño no obvias detrás de
`urgencias-core`. Cada sección es de tipo "qué se decidió, por qué, y
cómo retractarlo". No es exhaustivo — la mayoría del código se explica
solo; lo que está aquí vale la pena recordar.

---

## LOS, boarding, y qué nos dicen (y no nos dicen) los datos

El dataset visit-level provee timestamps de llegada y alta, pero no
separa el tiempo de workup clínico activo del tiempo de *boarding*
(paciente ya ingresado, esperando cama hospitalaria). Las distribuciones
históricas de LOS reflejan ambos efectos combinados, incluyendo la tasa
histórica de boarding del hospital durante el período de entrenamiento.
Esto significa:

1. Los pronósticos implícitamente asumen que los patrones futuros de
   boarding se parecen a los históricos.
2. Cambios estructurales en la disponibilidad de camas upstream —
   apertura de nuevas salas, cambios en políticas de admisión, cambios
   de capacidad a nivel hospitalario — degradarán la precisión hasta
   que el modelo se re-entrene sobre datos posteriores al cambio.
3. El estado de camas provisto por el usuario en la vista de surge es
   el mecanismo primario para sobrescribir las suposiciones históricas
   cuando las condiciones actuales son inusuales.

Una mejora futura sería ingerir el timestamp de decisión de admisión
desde Andesmed (cuando un médico de ER decide "este paciente sube") para
separar tiempo de workup del tiempo de boarding. Esto requiere o un
nuevo campo en el export del EMR, o un proxy derivado de timestamps de
entrada de órdenes. Ítem candidato de sprint 3 en `roadmap.md`.

---

## Por qué demo dual: sintético y DEIS

El repositorio incluye dos scripts de demo que muestran partes
complementarias del pipeline.

`scripts/demo_synthetic.py` corre el pipeline completo — carga la fixture
visit-level, computa la serie horaria, ajusta un forecaster de línea
base, y corre la simulación Monte Carlo. Produce tres PNG en `outputs/`.

`scripts/demo_deis.py` corre solo la capa de forecasting contra datos
reales públicos de DEIS MINSAL para Hospital de Puerto Montt y Hospital
de Frutillar. Produce tablas de backtesting y pronósticos forward a
6 meses. **No invoca el motor de simulación.**

La razón del split es honestidad arquitectónica. El dataset DEIS publica
conteos agregados por establecimiento-día-grupo de causa — no registros
visit-level con timestamps de llegada y alta. Esto tiene una consecuencia
directa:

- DEIS puede alimentar la **capa de forecasting** a grano diario o
  semanal. Sobre datos reales de hospitales chilenos. Fine.
- DEIS **no puede** alimentar el motor de simulación porque la
  simulación requiere distribuciones de LOS condicionales a agudeza y
  hora de llegada, que solo existen en datos visit-level.

Un único demo combinado tendría que o (a) simular sobre datos
fabricados, ocultando la dependencia de datos propietarios, o (b)
saltarse la simulación cuando corra sobre DEIS, rompiendo la simetría.
Dos demos explícitos hacen evidente qué parte del pipeline funciona con
qué tipo de datos, y qué traería un hospital que quiera operacionalizar
esto (datos visit-level).

## Manejo del período COVID en datos DEIS

Por defecto, `scripts/demo_deis.py` excluye los años 2020 y 2021 del
entrenamiento. La razón: la pandemia disruptó los patrones de atención
de urgencia de forma que no es representativa del régimen actual, y el
período post-pandemia (2022 a presente) ya provee 4+ años de datos
estables — suficiente para entrenar sin tener que modelar explícitamente
la disrupción COVID con una variable indicadora.

Esta decisión se puede sobrescribir cambiando la constante
`COVID_EXCLUDE_YEARS` al inicio de `demo_deis.py` o pasando un set vacío
si se agrega un flag en el futuro. Si se incluyen 2020–2021, la mayoría
de los modelos necesitarán una variable indicadora explícita para
capturar el break estructural.

Nota: los archivos DEIS 2017–2019 usan un formato diferente (xlsx + mdb
dentro del ZIP) que el parser CSV actual no soporta; el archivo DEIS
2020 tiene una fila de cabecera corrupta en la fuente. Incluso si se
revirtiera la exclusión por política, estos años solo estarán
disponibles tras implementar el soporte xlsx/mdb — tracked en
`roadmap.md`.

## Disclaimer metodológico

Para que quede registrado en el código y no solo en el README:

> Esta demostración usa datos públicos de DEIS MINSAL para mostrar el
> funcionamiento de las herramientas de forecasting sobre datos reales
> de hospitales chilenos. No constituye una evaluación operacional,
> clínica ni de calidad de los hospitales mencionados. Los hospitales
> se eligieron por ser referentes regionales en Los Lagos y por la
> disponibilidad pública de sus datos.

---

## Forecaster protocol: agnóstico del grano

`HorizonSpec.grain` acepta un alias de frecuencia de pandas arbitrario
(`"h"`, `"D"`, `"W-MON"`, `"ME"`, etc.). Las implementaciones del
protocolo son responsables de adaptarse internamente — por ejemplo,
`SeasonalNaiveBaseline` usa una llave estacional distinta por grano
(`(dow, hour)` para horario, `(month, dow)` para diario,
`(isoweek,)` para semanal, `(month,)` para mensual).

Esto significa que el mismo harness de evaluación sirve para los tres
horizontes de Eunosia (surge horario, roster mensual, anual) y para el
demo DEIS (semanal/diario), sin forks.

## LightGBM quantile regression: solo features de calendario

La versión en core usa features de calendario (hora, día de la semana,
mes, día del año, encodings sin/cos, banderas de fin de semana) y no
usa lags autoregresivos. Esta es una elección deliberada:

- El modelo es agnóstico del grano y del horizonte sin depender de una
  política de lag-awareness.
- Es honesto sobre la información disponible en tiempo de pronóstico:
  no existe leakage implícito.
- Hospitales que claan este código obtienen un modelo funcional sin
  supuestos sobre su estructura temporal.

La desventaja: no puede leveragear información reciente (último día,
última semana). Para horizontes cortos — en particular la decisión de
surge a 0–24 horas — los modelos con lags AR mejoran materialmente
sobre esta versión. Esos viven en `eunosia-forecast` (Horizon A).

## Regla de advertencia ≥5% en el harness

El harness (`eval/harness.py`) implementa una regla explícita: si el
mejor modelo candidato no supera al mejor baseline (SeasonalNaive o el
mejor de statsforecast) por al menos 5% en el cuantil de chequeo (P80
por defecto), se imprime una advertencia claramente visible a stdout.

Por qué existe: evita shippear un modelo complejo que no se gana su
complejidad. Una red TFT entrenada sobre 200k parámetros que empata a
AutoETS sobre pinball loss no debería entrar en producción aun si pasa
tests unitarios.

---

## Servidor de referencia (no es dashboard)

`urgencias_core.server` es server-side rendered, sin JS, sin build step.
Matplotlib renderiza PNGs a base64 inline. Las páginas imprimen y se
guardan como PDF correctamente desde cualquier navegador — esta es una
feature, no un accidente.

Esto es deliberadamente minimal. El dashboard interactivo bonito vive
en el repo privado `eunosia-forecast` y es una decisión comercial de
frontend separada. `urgencias-core/server` existe para que alguien que
clone el repo pueda abrir su navegador y ver qué hace el código sin
tener que leerlo.

## Ningún paquete PyPI

No hay versioning, changelog, citation.cff, ni publicación a PyPI.
`pyproject.toml` solo existe para que `uv sync` funcione localmente.

Razón: `urgencias-core` es **código de referencia**, no un paquete
mantenido. Prometer estabilidad de API crea obligaciones que no
queremos. Si alguien quiere depender de él, fork y pin a un commit.

---

## Zona horaria de los datos

Los parquets de entrada pueden ser tz-aware o tz-naive; el loader
normaliza a tz-naive (UTC). La fixture sintética es tz-naive porque
generar 3 años de datos horarios tz-aware en America/Santiago cruza
múltiples transiciones DST y produce timestamps "nonexistent" alrededor
de las fechas de cambio. Los exports reales de EMRs chilenos
históricamente también se publican tz-naive asumiendo hora local.
