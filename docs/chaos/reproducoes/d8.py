import hashlib, os
from chaos_common import *
from pathlib import Path
H="fighter_a,fighter_b,odds_a_decimal,odds_b_decimal,fonte,odds_a_tp,odds_b_tp\n"
def tree():
    out={}
    for d in ("data","docs"):
        for p in sorted((APP/d).rglob("*")):
            if p.is_file(): out[str(p.relative_to(APP))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out
def tentar(csv_text, nome_arq="card_teste.csv"):
    reset(); preditor_fixo(); sem_frescor()
    api_simulada([evento_api("Ana Alfa","Bia Beta",1.85,2.05),evento_api("Caio Gama","Davi Delta",1.55,2.6),evento_api("Eli Echo","Fabio Foxtrot",1.6,2.4),evento_api("Gil Golf","Hugo Hotel",1.7,2.3)])
    p=APP/"data/raw"/nome_arq; p.write_text(csv_text,encoding="utf-8")
    from src.card_report import load_card_odds
    # load direto
    try: df=load_card_odds(p); le=None
    except Exception as e: df=None; le=e
    antes=tree()
    try:
        publicar_card(csv=p, evento=EVENTO); pe=None
    except Exception as e: pe=e
    depois=tree()
    return df, le, pe, antes==depois, [k for k in depois if antes.get(k)!=depois[k]]
ok_total=True
L=["Ana Alfa,Bia Beta,1.80,2.10,ambas,,\n","Caio Gama,Davi Delta,1.50,2.70,ambas,,\n","Eli Echo,Fabio Foxtrot,1.60,2.40,ambas,,\n","Gil Golf,Hugo Hotel,1.70,2.30,ambas,,\n"]
def caso(nome, linhas, linha_esperada, extra=""):
    global ok_total
    df, le, pe, igual, mud = tentar(H+"".join(linhas))
    msg=str(le)
    cond = le is not None and pe is not None and f"linha {linha_esperada} " in msg and igual
    ok_total &= resultado(nome, cond, f"ValueError citando 'linha {linha_esperada}'; publicar tambem recusa; data/ e docs/ byte a byte iguais",
        f"load={type(le).__name__ if le else None}: {msg[:230]!r}; publicar={type(pe).__name__ if pe else None}; arvore_igual={igual} mudou={mud[:4]}")
# 1 linha do meio sem odds_b (arquivo linha 3)
l=list(L); l[1]="Caio Gama,Davi Delta,1.50,,ambas,,\n"; caso("D8a linha do meio sem odds_b", l, 3)
# 2a virgula decimal entre aspas
l=list(L); l[2]='Eli Echo,Fabio Foxtrot,1.60,"1,85",ambas,,\n'; caso("D8b odd \"1,85\" (entre aspas)", l, 4)
# 2b virgula decimal sem aspas (desloca colunas)
l=list(L); l[2]='Eli Echo,Fabio Foxtrot,1.60,1,85,ambas,,\n'; caso("D8b' odd 1,85 sem aspas (colunas deslocadas)", l, 4)
# 3 lutador vazio
l=list(L); l[1]="Caio Gama,,1.50,2.70,ambas,,\n"; caso("D8c lutador vazio", l, 3)
l=list(L); l[3]=",Hugo Hotel,1.70,2.30,ambas,,\n"; caso("D8c' lutador a vazio na ultima", l, 5)
# 4 branco intercalado + opcionais vazias: deve passar
txt = H + "\n" + L[0] + "\n\n" + L[1].replace(",ambas,,",",,,") + "\n" + L[2] + "\n" + L[3]
df, le, pe, igual, mud = tentar(txt)
cond = le is None and df is not None and len(df)==4 and pe is None
ok_total &= resultado("D8d linhas em branco intercaladas + colunas opcionais vazias", cond, "carrega 4 lutas; publicar nao recusa", f"load_err={le!r} n={None if df is None else len(df)} publicar_err={pe!r}")
# 4b erro depois de linhas em branco: linha certa
txt = H + "\n" + L[0] + "\n\nCaio Gama,Davi Delta,1.50,,ambas,,\n" + L[2]
df, le, pe, igual, mud = tentar(txt)
cond = le is not None and "linha 5 " in str(le) and igual
ok_total &= resultado("D8e erro apos linhas em branco cita linha do arquivo (5)", cond, "linha 5", f"{str(le)[:200]!r} arvore_igual={igual}")
print("D8:", "ok" if ok_total else "DEFEITO(S)")
