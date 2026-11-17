# E1: cabeza de salida en dos etapas

Pregunta: ¿se puede elegir el siguiente token leyendo solo una parte de la matriz de salida (748 MiB en Q8_0, el 23 % de lo que Gemma 4 26B-A4B
lee por token)? Resultado y lectura: ver `docs/RESULTS.md`; la salida en dos etapas se estimó en ≈ +12 % de generación y no está implementada.

## Pasos
1. Volcar estados finales reales con el motor propio (un prompt real por ejecución; los datos van fuera del repositorio):
   ```
   DAFO_SWARM_DUMP_FINAL_HIDDEN=DIR/states-NOMBRE.bin target/release/swarm-node model complete \
     --source MODELO.gguf --device metal:0 --max-new-tokens 160 --temperature 0 --prompt "$(cat PROMPT.txt)"
   ```
   Formato del volcado: por llamada, número de posiciones (`u32`) y sus estados normalizados (`f32`, 2.816 por posición). La primera llamada
   es el prompt; las siguientes, un token generado cada una.
2. Analizar (sin modelo cargado; usa ~1 GiB y un minuto):
   ```
   python3 analyze_screen.py --gguf MODELO.gguf --states 'DIR/states-*.bin' --out DIR/resultado.json [--bits 1,2,3,4] [--projections 128,256]
   ```

## Archivos
- `gguf_q8.py`: localiza la matriz de salida en el GGUF y la mapea como bloques Q8_0.
- `analyze_screen.py`: cribas por bloque (1-4 bits) y por proyección; puesto del token correcto, filas a leer por cota y bytes por token.

Memoria: medir la memoria libre antes de cada ejecución del modelo y no tener otro modelo cargado.
