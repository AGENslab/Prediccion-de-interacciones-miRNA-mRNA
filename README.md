# Predicción de Interacciones miRNA–mRNA en Humanos y Plantas mediante Deep Learning

Este repositorio contiene el código desarrollado para una tesis de título que implementa y evalúa modelos de Deep Learning orientados a predecir interacciones entre microRNAs (miRNAs) y genes mRNA, tanto en humanos como en especies vegetales. El enfoque combina arquitecturas CNN + Self-Attention y Transformers, entrenadas sobre secuencias biológicas reales y validadas mediante aprendizaje por transferencia.

## 📁 Estructura del repositorio

```
├── codigos/
│   ├── balancear_dataset.py
│   ├── Benchmark.py
│   ├── Benchmark_plantas.py
│   ├── buscar_pares.py
│   ├── combinar_dataset.py
│   ├── evaluar_modelo_final.py
│   ├── evaluar_modelo_finetune_plantas.py
│   ├── explorar_dataset.py
│   ├── finetune_cnn_attention_plantas.py
│   ├── finetune_transformer_rechazo_plantas.py
│   ├── modelo_cnn_keras.py
│   ├── modelo_tra_keras.py
│   ├── obtener_gen_mrna.py
│   ├── pla_agregar_secuencias.py
│   ├── pla_agregar_secuencias_transcrito.py
│   ├── pla_descargar_secuencias_plantas.py
│   ├── pla_encontrar_especies.py
│   ├── pla_filtrar_dataset_plantas.py
│   ├── pla_generar_combinaciones_nuevas.py
```

## 🧠 Modelos utilizados

- `modelo_cnn_attention_final.keras`: Arquitectura CNN con atención para detección de interacciones.
- `modelo_transformer_f.keras`: Modelo Transformer entrenado sobre secuencias de miRNA y mRNA.
- `modelo_finetune_transformer_plantas.keras*`: Scripts para adaptar modelos humanos a vegetales mediante *fine-tuning*.
- `modelo_finetune_rechazo_plantas.keras*`: Scripts para adaptar modelos humanos a vegetales mediante *fine-tuning*.

## 🔬 Principales etapas

1. **Preparación de datos**
   - `explorar_dataset.py`, `combinar_dataset.py`, `balancear_dataset.py`  
   - Extracción y codificación de secuencias desde TarBase, miRTarBase y NCBI.

2. **Entrenamiento de modelos**  
   - `modelo_cnn_keras.py`, `modelo_tra_keras.py`  
   - Entrenamiento supervisado con validación cruzada.

3. **Aprendizaje por transferencia (plantas)**  
   - `finetune_cnn_attention_plantas.py`  
   - `finetune_transformer_rechazo_plantas.py`  
   - Aplicación de *fine-tuning* sobre secuencias positivas vegetales.

4. **Evaluación y benchmark**  
   - `Benchmark.py`, `Benchmark_plantas.py`, `evaluar_modelo_final.py`  
   - Reportes de precisión, recall, F1-score, AUC y rechazo de falsos positivos.

5. **Manejo de secuencias biológicas**  
   - `obtener_gen_mrna.py`, `pla_agregar_secuencias.py`, etc.  
   - Scripts para extraer, mapear y codificar secuencias reales de NCBI y miRBase.

## 📊 Resultados destacados

- Accuracy en humanos: ~88% para ambos modelos.
- F1-score ponderado: 0.8793 (Transformer) vs. 0.8772 (CNN).
- Implementación de mecanismo de rechazo por error de reconstrucción para evitar falsos positivos.

## ⚙️ Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

## 👨‍🔬 Autor
Profesora guía
**Carol Moraga Quinteros** -
Alumno
**Jaime Escobar Gálvez**  
Tesis de Ingeniería Civil en Computación – Universidad de O’Higgins  
Año: 2025

## 📚 Citas y referencias

Basado en el uso de bases de datos públicas como [miRTarBase](https://mirtarbase.cuhk.edu.cn/) y [DIANA-TarBase](https://dianalab.e-ce.uth.gr/html/diana_tarbase.html). Secuencias extraídas desde [NCBI](https://www.ncbi.nlm.nih.gov/) y [miRBase](https://www.mirbase.org/).
