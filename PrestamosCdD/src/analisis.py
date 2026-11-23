"""
Funciones de apoyo del proyecto de préstamos.

Son pocas y cortas: solo evitan repetir los mismos gráficos y cálculos
en el cuaderno.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

AZUL = "#2E86AB"
ROJO = "#D1495B"


def configurar_estilo():
    """Estilo común de todas las gráficas."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 4)
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.titleweight"] = "bold"


def resumen_nulos(df):
    """Tabla con las columnas que tienen valores faltantes."""
    tabla = pd.DataFrame({
        "nulos": df.isna().sum(),
        "porcentaje": (df.isna().mean() * 100).round(1),
    })
    return tabla[tabla["nulos"] > 0].sort_values("nulos", ascending=False)


def graficar_numericas(df, columnas):
    """Cuatro histogramas en una sola figura (2 x 2)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    for ax, col in zip(axes.ravel(), columnas):
        sns.histplot(df[col].dropna(), bins=25, color=AZUL, ax=ax)
        ax.axvline(df[col].median(), color=ROJO, ls="--", lw=1.5,
                   label=f"Mediana: {df[col].median():.0f}")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def graficar_categoricas(df, columnas):
    """Barras horizontales con el porcentaje de cada categoría."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for ax, col in zip(axes.ravel(), columnas):
        conteo = df[col].fillna("Sin dato").value_counts()
        sns.barplot(x=conteo.values, y=conteo.index, color=AZUL, ax=ax)

        for i, valor in enumerate(conteo.values):
            ax.text(valor, i, f" {valor / len(df) * 100:.0f}%",
                    va="center", fontsize=8)

        ax.set_title(col)
        ax.set_xlabel("Solicitudes")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def tasa_aprobacion(df, columna):
    """Porcentaje de aprobación para cada categoría de una variable."""
    return (df.groupby(columna)["Aprobado"].mean() * 100).round(1)


def graficar_tasa_aprobacion(df, columna, ax):
    """Barra con la tasa de aprobación de cada categoría."""
    tasa = tasa_aprobacion(df, columna).sort_values(ascending=False)
    sns.barplot(x=tasa.index.astype(str), y=tasa.values, color=AZUL, ax=ax)
    ax.axhline(df["Aprobado"].mean() * 100, color=ROJO, ls="--",
               label=f"Promedio: {df['Aprobado'].mean() * 100:.0f}%")

    for i, valor in enumerate(tasa.values):
        ax.text(i, valor, f"{valor:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Aprobación según {columna}")
    ax.set_xlabel("")
    ax.set_ylabel("% aprobado")
    ax.legend(fontsize=8)


def evaluar(nombre, modelo, X_train, y_train, X_test, y_test):
    """Calcula las métricas principales en entrenamiento y prueba."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    proba_train = modelo.predict_proba(X_train)[:, 1]
    proba_test = modelo.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    return {
        "modelo": nombre,
        "auc_train": roc_auc_score(y_train, proba_train),
        "auc_test": roc_auc_score(y_test, proba_test),
        "exactitud": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test),
        "recall": recall_score(y_test, pred_test),
        "f1": f1_score(y_test, pred_test),
    }


def tabla_resultados(resultados):
    """Ordena los resultados y calcula la brecha entre train y test."""
    tabla = pd.DataFrame(resultados)
    tabla["brecha"] = tabla["auc_train"] - tabla["auc_test"]
    columnas = ["modelo", "auc_train", "auc_test", "brecha",
                "exactitud", "precision", "recall", "f1"]
    return tabla[columnas].round(3).sort_values("auc_test", ascending=False)


def graficar_roc(modelos, X_test, y_test):
    """Curvas ROC de varios modelos en una sola figura."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))

    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2,
                label=f"{nombre} (AUC = {roc_auc_score(y_test, proba):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.50)")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def graficar_train_test(tabla):
    """Barras comparando AUC de entrenamiento y de prueba."""
    fig, ax = plt.subplots(figsize=(9, 4))
    posiciones = np.arange(len(tabla))
    ancho = 0.35

    ax.bar(posiciones - ancho / 2, tabla["auc_train"], ancho,
           label="Entrenamiento", color=AZUL)
    ax.bar(posiciones + ancho / 2, tabla["auc_test"], ancho,
           label="Prueba", color=ROJO)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(tabla["modelo"], rotation=15, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("ROC AUC")
    ax.set_title("¿El modelo memoriza o generaliza?")
    ax.legend()
    plt.tight_layout()
    plt.show()


def graficar_correlacion(df, columnas):
    """Mapa de calor con la correlación de Spearman."""
    correlacion = df[columnas].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlacion, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlación entre variables (Spearman)")
    plt.tight_layout()
    plt.show()

    return correlacion


def explicar_caso(modelo, X_caso, top=8):
    """Descompone la predicción: valor base + suma de coeficiente x valor escalado."""
    imputador = modelo.named_steps["imputar"]
    escalador = modelo.named_steps["escalar"]
    logistica = modelo.named_steps["modelo"]

    x = escalador.transform(imputador.transform(X_caso))[0]
    contribuciones = logistica.coef_[0] * x

    tabla = pd.DataFrame({
        "variable": X_caso.columns,
        "valor": X_caso.iloc[0].values,
        "contribucion": contribuciones,
    })
    tabla = tabla.loc[tabla["contribucion"].abs().nlargest(top).index]

    base = logistica.intercept_[0]
    log_odds = base + contribuciones.sum()
    probabilidad = modelo.predict_proba(X_caso)[0, 1]

    return tabla, base, log_odds, probabilidad
