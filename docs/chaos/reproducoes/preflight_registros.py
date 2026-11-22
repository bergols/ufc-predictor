"""
Pre-voo da instalacao: os registros REAIS da arvore operacional passam nas
regras de leitura do codigo novo?

    python docs/chaos/reproducoes/preflight_registros.py <arvore_operacional>

Roda a partir de um clone no SHA a instalar (o codigo importado e o do
clone) e so LE os arquivos de <arvore_operacional>/data. Nao grava nada, nao
cria trava, nao usa rede.

Confere, para cada registro que existir:
  - validar_csv (termina em \\n, campos por linha = cabecalho, aspas fechadas);
  - cabecalho igual as colunas que o codigo novo usa para ANEXAR
    (snapshots, chamadas, cotacoes, medianas, decisoes, sombra) -- divergencia
    aqui faria a captura/publicacao recusar a gravacao depois da instalacao;
  - prediction_history.csv carregavel pelo leitor do codigo novo e com todo
    prediction_id apontando para versao legivel.
Sai com codigo 1 se algo reprovar, listando o arquivo e o motivo.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ_CLONE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ_CLONE))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    alvo = Path(sys.argv[1]).resolve()
    dados = alvo / "data"
    if not dados.is_dir():
        print(f"REPROVADO: {dados} nao existe")
        return 1
    if alvo == RAIZ_CLONE:
        print("REPROVADO: rode a partir de um clone separado, nao da arvore operacional")
        return 1

    from src import decisoes, mediana, proveniencia, snapshots, sombra
    from src.persistencia import RegistroIncompleto, validar_csv
    from src.prediction_history import _load_raw

    com_colunas = {
        "previsoes_snapshots.csv": snapshots.COLUNAS,
        "proveniencia/chamadas.csv": proveniencia._COLUNAS,
        "proveniencia/cotacoes.csv": proveniencia._COLUNAS_COT,
        "proveniencia/medianas.csv": mediana.COLUNAS,
        "decisoes.csv": decisoes.COLUNAS,
        "experimentos/sombra/sombra_previsoes.csv": sombra.COLUNAS,
        "experimentos/sombra/sombra_resultados.csv": sombra.COLUNAS_RESULTADO,
    }
    # snapshots_mercado.csv fica de fora: nenhum codigo o le ou grava.
    so_integridade = ["prediction_history.csv", "odds_template.csv",
                      "raw/upcoming_card_odds.csv"]

    falhas, ok = [], 0
    for rel in list(com_colunas) + so_integridade:
        caminho = dados / rel
        if not caminho.exists():
            print(f"  -    {rel}: ausente (ok, o codigo cria quando precisar)")
            continue
        try:
            validar_csv(caminho)
        except RegistroIncompleto as exc:
            falhas.append(f"{rel}: {exc}")
            continue
        if caminho.stat().st_size == 0:
            print(f"  ok   {rel}: vazio (o codigo grava o cabecalho na primeira linha)")
            ok += 1
            continue
        if rel in com_colunas:
            # Mesma leitura do codigo novo (persistencia.anexar_csv): utf-8 sem
            # tratar BOM e csv.reader sem strip. Comparar mais frouxo que isso
            # aprovaria um cabecalho que o codigo depois recusa.
            import csv
            with caminho.open(encoding="utf-8", newline="") as fh:
                cab = tuple(next(csv.reader(fh), []))
            if cab != tuple(com_colunas[rel]):
                # decisoes.csv tem alinhamento proprio de cabecalho antigo
                if rel == "decisoes.csv" and set(cab) <= set(com_colunas[rel]):
                    print(f"  ok   {rel}: cabecalho antigo, alinhado pelo proprio decisoes.registrar")
                    ok += 1
                    continue
                falhas.append(f"{rel}: cabecalho {cab} difere das colunas do codigo {tuple(com_colunas[rel])}")
                continue
        print(f"  ok   {rel}")
        ok += 1

    hist = dados / "prediction_history.csv"
    if hist.exists() and not any(f.startswith("prediction_history.csv") for f in falhas):
        try:
            df = _load_raw(hist)
            arq = dados / "previsoes_snapshots.csv"
            sem = []
            for pid in df["prediction_id"].dropna().astype(str):
                if snapshots.versao(pid.strip(), arq) is None:
                    sem.append(pid)
            if sem:
                falhas.append(f"prediction_history.csv: {len(sem)} prediction_id sem versao legivel (ex.: {sem[:3]})")
            else:
                print(f"  ok   prediction_history.csv: {len(df)} linhas legiveis; todo prediction_id resolve")
        except Exception as exc:  # noqa: BLE001
            falhas.append(f"prediction_history.csv: leitor do codigo novo falhou: {exc}")

    print(f"\nRegistros aprovados: {ok} | reprovados: {len(falhas)}")
    for f in falhas:
        print(f"REPROVADO: {f}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
