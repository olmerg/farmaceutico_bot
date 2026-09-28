# Farmacéutico Bot — Fase 0

Agente con LangChain + NVIDIA Build API que consulta un dataset de medicamentos del INVIMA.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env-example .env
# Edita .env con tu NVIDIA_API_KEY
```

## Estructura

```
data/medicamentos.csv   # 5000 registros (sample del Excel original, semilla 42)
src/tools.py            # 5 tools de consulta
src/agente.py           # Agente LangChain + NVIDIA
```

## Ejecutar

```powershell
.\.venv\Scripts\python src\agente.py
```

Modo interactivo. Escribe preguntas sobre medicamentos o `salir` para terminar.

## Tools disponibles

| Tool                            | Qué hace                          | Ejemplo                                    |
| ------------------------------- | ---------------------------------- | ------------------------------------------ |
| `buscar_por_principio_activo` | Filtra por principio activo        | "¿Qué medicamentos contienen ACICLOVIR?" |
| `buscar_por_nombre_comercial` | Filtra por nombre comercial        | "¿Qué productos hay de TYLENOL?"         |
| `consultar_vencidos`          | Lista medicamentos vencidos        | "¿Cuáles están vencidos?"               |
| `consultar_por_titular`       | Filtra por laboratorio             | "¿Qué tiene Pfizer?"                     |
| `consultar_por_via`           | Filtra por vía de administración | "¿Cuáles son orales?"                    |
