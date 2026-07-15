"""
Configuracoes centrais do projeto UFC Predictor.

Mantem todos os caminhos e constantes num unico lugar para facilitar
manutencao e para que os outros modulos nao tenham paths/numeros
"hardcoded" espalhados pelo codigo.
"""
from pathlib import Path

# Raiz do projeto (pasta que contem este arquivo)
PROJECT_ROOT = Path(__file__).resolve().parent

# Diretorios de dados
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Arquivos de dados brutos (formato "scrape" do UFCStats.com)
RAW_FIGHTS_CSV = RAW_DIR / "fights.csv"
RAW_FIGHT_STATS_CSV = RAW_DIR / "fight_stats.csv"
RAW_FIGHTERS_CSV = RAW_DIR / "fighters.csv"

# Arquivo de dados brutos (formato "dataset publico", usado como fallback)
PUBLIC_DATASET_CSV = RAW_DIR / "public_dataset.csv"

# Diretorio onde ficam os 6 CSVs baixados do espelho GitHub
# (Greco1899/scrape_ufc_stats), antes da conversao para o formato canonico
GITHUB_MIRROR_DIR = RAW_DIR / "github_mirror"

SQLITE_DB_PATH = RAW_DIR / "ufc_data.sqlite"

# Arquivos de dados processados
FEATURES_CSV = PROCESSED_DIR / "fight_features.csv"

# Template para o usuario preencher odds de mercado manualmente
ODDS_TEMPLATE_CSV = DATA_DIR / "odds_template.csv"

# Historico de previsoes por evento (paper trading): previsoes CONGELADAS
# no pre-registro do card + resultados sincronizados do odds_template.
# Ver src/prediction_history.py.
PREDICTION_HISTORY_CSV = DATA_DIR / "prediction_history.csv"

# Odds historicas reais (dataset jansen88/ufc-data, betmma.tips) -- ver src/market_odds.py
MARKET_ODDS_CSV = RAW_DIR / "market_odds.csv"
MARKET_COMPARISON_CSV = PROCESSED_DIR / "market_comparison.csv"

# Artefatos de modelo (vencedor)
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_calibrated.joblib"
GBM_MODEL_PATH = MODELS_DIR / "gbm_calibrated.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"

# Artefatos de modelo (metodo de vitoria e round de finalizacao)
METHOD_LOGREG_MODEL_PATH = MODELS_DIR / "method_logreg_calibrated.joblib"
METHOD_GBM_MODEL_PATH = MODELS_DIR / "method_gbm_calibrated.joblib"
ROUND_LOGREG_MODEL_PATH = MODELS_DIR / "round_logreg_calibrated.joblib"
ROUND_GBM_MODEL_PATH = MODELS_DIR / "round_gbm_calibrated.joblib"
METHOD_METADATA_PATH = MODELS_DIR / "method_training_metadata.json"

# Verificacao de frescor dos dados: avisa se o evento mais recente nos dados
# coletados estiver mais velho que isso. O UFC tem eventos quase toda semana,
# entao um gap acima de ~2 semanas quase sempre significa fonte estagnada
# (ja aconteceu: o job do espelho GitHub parou em mai/2026 sem aviso).
DATA_FRESHNESS_MAX_GAP_DAYS = 14

# Parametros de negocio / modelagem
RANDOM_SEED = 42
N_RECENT_FIGHTS = 5                  # janela para "forma recente"
MIN_FIGHTS_FOR_RELIABLE_STATS = 3    # abaixo disso, marca flag de "dados insuficientes"

# Rating Elo (ver src/ratings.py). K controla a velocidade de adaptacao do
# rating a cada resultado. Estreantes comecam no rating base.
# K=64 escolhido por validacao em cal_select (grade 16/24/32/40/64 via
# `python -m src.tuning`, jul/2026). Ganho sobre K=32 foi MARGINAL
# (log loss medio 0.6723 vs 0.6725 em cal_select) -- a grade toda variou
# so ~0.0015, entao K quase nao importa nessa faixa.
ELO_K_FACTOR = 64.0
ELO_BASE_RATING = 1500.0
# Multiplicadores de K por decisividade da vitoria (margem por metodo).
# None = Elo simples. Testado em cal_select (jul/2026, K=64): NENHUM esquema
# de margem bateu o Elo simples (sem-margem 0.6723 vs leve 0.6737, agressivo
# 0.6728, so-bonus 0.6742), entao producao usa None. O parametro existe em
# compute_elo_ratings para experimentos futuros (ver src/tuning.py).
ELO_METHOD_MULTIPLIERS = None

# Split temporal: fracoes dos dados (ordenados por data) usadas para cada etapa.
# Ex.: treina com os 70% mais antigos, calibra com os proximos 15%,
# testa com os 15% mais recentes (nunca vistos ate a avaliacao final).
# Isso evita vazamento temporal (nunca usar o "futuro" para prever o "passado").
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.15

assert abs(TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION - 1.0) < 1e-9

# User-agent "educado" para o scraper (identifica o bot, uso nao comercial)
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (compatible; ufc-predictor-research-bot/1.0) "
    "uso pessoal e nao comercial"
)
SCRAPER_DELAY_SECONDS = 1.0  # intervalo entre requisicoes, para nao sobrecarregar o site
