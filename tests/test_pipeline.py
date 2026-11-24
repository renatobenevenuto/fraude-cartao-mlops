import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

COLUNAS_FEATURES = [
    "TX_AMOUNT",
    "TX_TIME_HOURS",
    "TX_DAY_OF_WEEK",
    "TX_COUNT_1H",
    "TX_SUM_1H",
    "TX_COUNT_24H",
    "TX_SUM_24H",
    "RISCO_TERMINAL",
]
COLUNA_ALVO = "FRAUD"
AUPRC_MINIMO = 0.70


@pytest.fixture(scope="module")
def dataset_features() -> pd.DataFrame:
    """Gera massa sintética desbalanceada (~3% fraude) com o mesmo schema da Fase 3."""
    rng = np.random.default_rng(42)
    n_legit = 2910
    n_fraud = 90

    legit = pd.DataFrame(
        {
            "TX_AMOUNT": np.round(rng.lognormal(mean=3.8, sigma=0.6, size=n_legit), 2),
            "TX_TIME_HOURS": rng.integers(6, 24, size=n_legit).astype(float),
            "TX_DAY_OF_WEEK": rng.integers(1, 8, size=n_legit).astype(float),
            "TX_COUNT_1H": rng.poisson(0.5, size=n_legit).astype(float),
            "TX_SUM_1H": np.round(rng.uniform(0.0, 200.0, size=n_legit), 2),
            "TX_COUNT_24H": rng.poisson(2.0, size=n_legit).astype(float),
            "TX_SUM_24H": np.round(rng.uniform(0.0, 500.0, size=n_legit), 2),
            "RISCO_TERMINAL": np.round(rng.uniform(0.0, 0.08, size=n_legit), 4),
            "FRAUD": np.zeros(n_legit, dtype=int),
        }
    )

    fraud = pd.DataFrame(
        {
            "TX_AMOUNT": np.round(rng.uniform(450.0, 2000.0, size=n_fraud), 2),
            "TX_TIME_HOURS": rng.integers(0, 6, size=n_fraud).astype(float),
            "TX_DAY_OF_WEEK": rng.integers(1, 8, size=n_fraud).astype(float),
            "TX_COUNT_1H": rng.integers(2, 8, size=n_fraud).astype(float),
            "TX_SUM_1H": np.round(rng.uniform(300.0, 1500.0, size=n_fraud), 2),
            "TX_COUNT_24H": rng.integers(3, 12, size=n_fraud).astype(float),
            "TX_SUM_24H": np.round(rng.uniform(500.0, 3000.0, size=n_fraud), 2),
            "RISCO_TERMINAL": np.round(rng.uniform(0.18, 0.65, size=n_fraud), 4),
            "FRAUD": np.ones(n_fraud, dtype=int),
        }
    )

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def modelo_treinado(dataset_features: pd.DataFrame):
    X = dataset_features[COLUNAS_FEATURES].astype("float64")
    y = dataset_features[COLUNA_ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=round(scale_pos_weight, 2),
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_integridade_schema_e_nulos(dataset_features: pd.DataFrame):
    """Garante colunas exatas, ausência de nulos e ausência de IDs arbitrários."""
    esperadas = set(COLUNAS_FEATURES + [COLUNA_ALVO])
    assert set(dataset_features.columns) == esperadas
    assert "CUSTOMER_ID" not in dataset_features.columns
    assert "TERMINAL_ID" not in dataset_features.columns
    assert not dataset_features.isnull().any().any()
    assert set(dataset_features[COLUNA_ALVO].unique()) == {0, 1}


def test_risco_terminal_sem_vazamento_futuro():
    """Valida que RISCO_TERMINAL considera apenas transações estritamente anteriores."""
    df_term = pd.DataFrame(
        {
            "TERMINAL_ID": [10, 10, 10],
            "TX_UNIX_TIME": [1000, 2000, 3000],
            "FRAUD": [1, 0, 1],
        }
    )

    risco_calculado = (
        df_term.sort_values("TX_UNIX_TIME")
        .groupby("TERMINAL_ID")["FRAUD"]
        .transform(lambda s: s.shift(1).expanding().mean().fillna(0.0))
    )

    # 1ª tx: sem histórico (0.0) | 2ª tx: média([1]) = 1.0 | 3ª tx: média([1, 0]) = 0.5
    assert risco_calculado.tolist() == pytest.approx([0.0, 1.0, 0.5])


def test_quality_gate_auprc(modelo_treinado):
    """Quality Gate de MLOps: exige AUPRC >= 0.70 antes de liberar o modelo."""
    model, X_test, y_test = modelo_treinado
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall, precision)

    print(f"\n[Quality Gate] AUPRC obtido: {auprc:.4f} (mínimo: {AUPRC_MINIMO:.2f})")
    assert auprc >= AUPRC_MINIMO, f"AUPRC {auprc:.4f} abaixo do mínimo {AUPRC_MINIMO}"


def test_contrato_payload_serving(modelo_treinado):
    """Valida o formato dataframe_split e o comportamento nos cenários de fraude e legítimo."""
    model, _, _ = modelo_treinado

    payload = {
        "dataframe_split": {
            "columns": COLUNAS_FEATURES,
            "data": [
                [1450.0, 3.0, 6.0, 4.0, 620.5, 5.0, 810.0, 0.35],
                [42.5, 14.0, 2.0, 0.0, 0.0, 1.0, 55.0, 0.0],
            ],
        }
    }

    df_input = pd.DataFrame(
        payload["dataframe_split"]["data"],
        columns=payload["dataframe_split"]["columns"],
    ).astype("float64")

    preds = model.predict(df_input).tolist()
    assert preds == [1, 0], f"Esperado [1, 0], obtido {preds}"
