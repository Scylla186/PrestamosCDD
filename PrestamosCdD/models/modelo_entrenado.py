"""
Modelo de aprobación de préstamos en Python puro (sin scikit-learn ni joblib).

Reescritura de `modelo_entrenado.joblib`, que era un Pipeline de scikit-learn:
    SimpleImputer(strategy="median")
    -> StandardScaler()
    -> LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=42)

Los números de abajo se extrajeron del modelo ya entrenado, así que da las
mismas probabilidades que el archivo original.

Uso rápido:
    from modelo_entrenado import predecir
    resultado = predecir({
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
        "LoanAmount": 130, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Property_Area": "Semiurban",
    })
    print(resultado)
"""

import math

# Orden exacto de las columnas con que se entrenó el modelo.
COLUMNAS = ['Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Ingreso_Total', 'Gender_Male', 'Gender_Sin dato', 'Married_Sin dato', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Sin dato', 'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']

# Paso 1 · SimpleImputer: valor (mediana del entrenamiento) para rellenar datos faltantes.
MEDIANAS = [
    0.0,  # Dependents
    128.0,  # LoanAmount
    360.0,  # Loan_Amount_Term
    1.0,  # Credit_History
    5391.0,  # Ingreso_Total
    1.0,  # Gender_Male
    0.0,  # Gender_Sin dato
    0.0,  # Married_Sin dato
    1.0,  # Married_Yes
    0.0,  # Education_Not Graduate
    0.0,  # Self_Employed_Sin dato
    0.0,  # Self_Employed_Yes
    0.0,  # Property_Area_Semiurban
    0.0,  # Property_Area_Urban
]

# Paso 2 · StandardScaler: (x - media) / desviación.
MEDIAS = [
    0.7494908350305499,  # Dependents
    146.52342158859472,  # LoanAmount
    341.74338085539716,  # Loan_Amount_Term
    0.8635437881873728,  # Credit_History
    7099.535234191038,  # Ingreso_Total
    0.8004073319755601,  # Gender_Male
    0.02240325865580448,  # Gender_Sin dato
    0.006109979633401222,  # Married_Sin dato
    0.6496945010183299,  # Married_Yes
    0.22606924643584522,  # Education_Not Graduate
    0.054989816700611,  # Self_Employed_Sin dato
    0.13441955193482688,  # Self_Employed_Yes
    0.37474541751527496,  # Property_Area_Semiurban
    0.329938900203666,  # Property_Area_Urban
]

DESVIACIONES = [
    1.0012084070541567,  # Dependents
    85.29904840716861,  # LoanAmount
    65.04160764484553,  # Loan_Amount_Term
    0.3432723613551995,  # Credit_History
    6685.93862748499,  # Ingreso_Total
    0.39969417670930063,  # Gender_Male
    0.14799105600476534,  # Gender_Sin dato
    0.07792719539596328,  # Married_Sin dato
    0.477065568203023,  # Married_Yes
    0.41828452308419717,  # Education_Not Graduate
    0.22796038418954334,  # Self_Employed_Sin dato
    0.341102530029414,  # Self_Employed_Yes
    0.4840571139510061,  # Property_Area_Semiurban
    0.470190623402957,  # Property_Area_Urban
]

# Paso 3 · Regresión logística: un coeficiente por columna y un intercepto.
COEFICIENTES = [
    0.03731112913747059,  # Dependents
    -0.019719226664025386,  # LoanAmount
    0.013225670800326298,  # Loan_Amount_Term
    0.5273558384821319,  # Credit_History
    -0.036909077618922266,  # Ingreso_Total
    -0.015052770737529485,  # Gender_Male
    -0.001965602182427181,  # Gender_Sin dato
    0.06706659736917749,  # Married_Sin dato
    0.09806983613092143,  # Married_Yes
    -0.08397128636909143,  # Education_Not Graduate
    -0.011577630034131457,  # Self_Employed_Sin dato
    -0.023740985009519724,  # Self_Employed_Yes
    0.15536666280604075,  # Property_Area_Semiurban
    -0.012511117977336856,  # Property_Area_Urban
]

INTERCEPTO = 0.09822625647577071

UMBRAL = 0.5  # probabilidad >= UMBRAL -> "Aprobado"


def _es_faltante(valor):
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def probabilidad_aprobado(fila):
    """Devuelve la probabilidad (0 a 1) de aprobar.

    `fila` es un diccionario {nombre_de_columna: valor} con las 14 columnas
    de COLUMNAS (ya transformadas). Si falta una o es None/NaN, se usa la mediana.
    """
    z = INTERCEPTO
    for i, col in enumerate(COLUMNAS):
        x = fila.get(col)
        if _es_faltante(x):
            x = MEDIANAS[i]
        x_escalado = (x - MEDIAS[i]) / DESVIACIONES[i]
        z += COEFICIENTES[i] * x_escalado
    return 1.0 / (1.0 + math.exp(-z))


def preparar_solicitud(s):
    """Convierte una solicitud "cruda" (como viene en el CSV) a las 14 columnas del modelo.

    Repite la limpieza de los cuadernillos 1 a 3: "3+" -> 3, categorías vacías
    -> "Sin dato", Ingreso_Total = solicitante + codeudor, y variables
    dummy (la categoría omitida es Female, No, Graduate y Rural).
    """
    def categoria(nombre):
        v = s.get(nombre)
        return "Sin dato" if _es_faltante(v) or v == "" else v

    dependientes = s.get("Dependents")
    if _es_faltante(dependientes) or dependientes == "":
        dependientes = None
    else:
        dependientes = float(str(dependientes).replace("3+", "3"))

    ingreso = float(s.get("ApplicantIncome") or 0) + float(s.get("CoapplicantIncome") or 0)
    genero, casado, indep = categoria("Gender"), categoria("Married"), categoria("Self_Employed")
    zona = s.get("Property_Area")

    return {
        "Dependents": dependientes,
        "LoanAmount": s.get("LoanAmount"),
        "Loan_Amount_Term": s.get("Loan_Amount_Term"),
        "Credit_History": s.get("Credit_History"),
        "Ingreso_Total": ingreso,
        "Gender_Male": int(genero == "Male"),
        "Gender_Sin dato": int(genero == "Sin dato"),
        "Married_Sin dato": int(casado == "Sin dato"),
        "Married_Yes": int(casado == "Yes"),
        "Education_Not Graduate": int(s.get("Education") == "Not Graduate"),
        "Self_Employed_Sin dato": int(indep == "Sin dato"),
        "Self_Employed_Yes": int(indep == "Yes"),
        "Property_Area_Semiurban": int(zona == "Semiurban"),
        "Property_Area_Urban": int(zona == "Urban"),
    }


def predecir(solicitud):
    """Predice a partir de una solicitud cruda. Devuelve un diccionario con el resultado."""
    p = probabilidad_aprobado(preparar_solicitud(solicitud))
    return {
        "probabilidad_aprobado": round(p, 4),
        "decision": "Aprobado" if p >= UMBRAL else "Rechazado",
    }


if __name__ == "__main__":
    ejemplo = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
        "LoanAmount": 130, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Property_Area": "Semiurban",
    }
    print(predecir(ejemplo))
