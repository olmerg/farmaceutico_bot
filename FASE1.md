# Fase 1 — Evaluación del Agente Farmacéutico

## Objetivo

Validar de forma objetiva si el agente funciona correctamente usando métricas de machine learning.

## Conceptos clave

### 1. Ground Truth
Respuestas esperadas que sirven como referencia para comparar contra lo que produce el agente.

### 2. Métricas de evaluación

| Métrica | Definición | Cómo se calcula |
|---------|------------|-----------------|
| **Tool Precision** | % de veces que el agente llamó a la tool correcta | `llamadas_correctas / total_llamadas` |
| **Tool Recall** | % de resultados correctos que el agente encontró | `resultados_encontrados / resultados_esperados` |
| **Exactitud** | % de respuestas que contienen datos correctos del dataset | `respuestas_correctas / total_respuestas` |
| **Consistencia** | % de veces que el agente responde igual en N ejecuciones | `respuestas_identicas / total_ejecuciones` |
| **Latencia** | Tiempo promedio de respuesta | `suma_tiempo / total_respuestas` |

### 3. Tipos de pruebas

| Tipo | Qué valida | Ejemplo |
|------|------------|---------|
| **Tool correcta** | ¿Llamó a la tool adecuada? | Preguntar por principio activo → `buscar_por_principio_activo` |
| **Argumentos correctos** | ¿Pasó los argumentos correctos? | `{"principio": "ACICLOVIR"}` |
| **Resultado correcto** | ¿Encontró los medicamentos correctos? | Verificar contra el dataset |
| **Consistencia** | ¿Responde igual siempre? | Ejecutar 3 veces la misma pregunta |

## Estructura de la evaluación

```
data/
  evaluacion.json          # Preguntas con ground truth
src/
  evaluador.py             # Código de evaluación
resultados/
  metricas.csv             # Resultados
```

## Pasos

1. **Definir ground truth**: Crear preguntas con respuesta esperada
2. **Ejecutar agente**: Correr el agente sobre las preguntas
3. **Calcular métricas**: Comparar respuestas contra ground truth
4. **Analizar resultados**: Identificar fallos y patrones

## Criterios de éxito

- Tool Precision > 90%
- Tool Recall > 85%
- Exactitud > 80%
- Consistencia > 95%

## Reto: Comparar NVIDIA vs Groq

Cambiar el modelo al disponible gratis en Groq y comparar resultados.

### Setup

```powershell
# Agregar clave de Groq al .env
GROQ_API_KEY=tu_clave_aqui
GROQ_MODEL=llama-3.3-70b-versatile
```

### Qué comparar

| Métrica | NVIDIA | Groq |
|---------|--------|------|
| Tool Precision | ? | ? |
| Exactitud | ? | ? |
| Consistencia | ? | ? |
| Latencia | ? | ? |

### Preguntas para analizar

- ¿Cuál modelo tiene mejor tool precision?
- ¿Cuál responde más rápido?
- ¿Cuál es más consistente?
- ¿Vale la pena pagar por uno o el gratis es suficiente?
