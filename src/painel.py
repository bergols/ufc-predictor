"""
src/painel.py

A pagina publicada, no formato "ficha de confronto": o card na ordem da
transmissao, uma faixa espelhada por luta, detalhes de auditoria sob demanda,
historico e metodologia na mesma pagina.

So DESENHA. Recebe a analise que `analyze_card` ja montou e le o que ja esta
gravado -- versoes em previsoes_snapshots.csv, o historico e as cotacoes em
proveniencia/cotacoes.csv -- para mostrar horarios e referencias. Nunca
consulta odds, recalcula previsao ou escreve registro. Dado ausente aparece
como indisponivel; nada e reconstruido com dado de hoje.

Pecas reaproveitadas de proposito, porque ja foram aprovadas e tem teste:
o eixo unico com o vao de divergencia (`card_report._gap_bar`), a nota das
duas odds, as notas por luta (`_matched_note`), o retrato
(`prediction_history.avatar_html`) e o painel de historico inteiro
(`render_history_panel`), que traz os alarmes de fechamento e de registro.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone

import pandas as pd

import config

BRT = timezone(timedelta(hours=-3), "BRT")   # Brasilia, sem horario de verao
INDISPONIVEL = "indisponível"


def _e(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _pct(p) -> str:
    return INDISPONIVEL if p is None else f"{p * 100:.1f}%"


def _odd(x) -> str:
    return INDISPONIVEL if not x else f"{float(x):.2f}"


def _quando(iso: str, *, data: bool = True) -> str:
    if not iso:
        return INDISPONIVEL
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(BRT)
    except ValueError:
        return INDISPONIVEL
    return t.strftime("%d/%m %H:%M") if data else t.strftime("%H:%M")


def _curto(txt: str) -> str:
    """Na faixa estreita, 'indisponivel' vira 'indisp.', com o texto inteiro
    para leitor de tela e no title."""
    if txt != INDISPONIVEL:
        return txt
    return (f'<abbr class="nd" aria-hidden="true" title="{INDISPONIVEL}">indisp.</abbr>'
            f'<span class="sr">{INDISPONIVEL}</span>')


# ---------------------------------------------------------------------------
# leitura do que ja esta gravado (somente leitura)
# ---------------------------------------------------------------------------

def _ler(caminho) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def proveniencia_do_evento(event_date: str) -> dict:
    """{frozenset(par): {...}}: versao, horarios, referencias e bloqueios."""
    if not event_date:
        return {}
    hist = _ler(config.PREDICTION_HISTORY_CSV)
    if hist.empty or "event_date" not in hist.columns:
        return {}
    snaps = _ler(config.DATA_DIR / "previsoes_snapshots.csv")
    cot = _ler(config.DATA_DIR / "proveniencia" / "cotacoes.csv")
    capt = dict(zip(cot.get("ref", []), cot.get("captured_at", [])))
    out = {}
    for _, r in hist[hist["event_date"] == event_date].iterrows():
        pid = r.get("prediction_id", "")
        v = None
        if pid and not snaps.empty and "prediction_id" in snaps.columns:
            achadas = snaps[snaps["prediction_id"] == pid]
            v = achadas.iloc[0] if len(achadas) == 1 else None
        g = lambda k: "" if v is None else str(v.get(k, ""))  # noqa: E731
        out[frozenset((r["fighter_a"], r["fighter_b"]))] = {
            "prediction_id": pid,
            "criado_em": g("criado_em"),
            "inicio": g("inicio_programado"),
            "manifesto": g("manifesto_sha256"),
            "model_name": g("model_name") or r.get("model_name", ""),
            "fonte_card": r.get("fonte_card", ""),
            "casamento": r.get("casamento", ""),
            "bloqueio": r.get("bloqueio_cotacao", ""),
            "ref_tp_a": r.get("ref_tp", ""), "ref_tp_b": r.get("ref_tp_b", ""),
            "tp_lido_em": capt.get(r.get("ref_tp", ""), ""),
            "sharp_prob": r.get("sharp_prob", ""),
            "sharp_ref": r.get("sharp_ref", ""),
            "sharp_lido_em": capt.get(r.get("sharp_ref", ""), ""),
        }
    return out


# ---------------------------------------------------------------------------
# frescor
# ---------------------------------------------------------------------------

def frescor_html(gap) -> str:
    from src.card_report import FRESCOR_NAO_VERIFICADO
    if gap == FRESCOR_NAO_VERIFICADO:
        # NAO e o mesmo que falhar ao verificar: em somente_render a base nem e
        # consultada, e dizer "nao foi possivel" inventaria um problema.
        return ('<div class="status"><span class="dot"></span>Frescor da base não '
                'verificado: esta página foi redesenhada a partir das versões de '
                'previsão já gravadas, sem consultar a base.</div>')
    if gap is None:
        return ('<div class="status bad"><span class="dot"></span>Não foi possível '
                'verificar o frescor da base de dados.</div>')
    if gap > config.DATA_FRESHNESS_MAX_GAP_DAYS:
        return (f'<div class="status bad"><span class="dot"></span>DADOS DESATUALIZADOS: '
                f'o evento mais recente na base tem {gap} dias (limite: '
                f'{config.DATA_FRESHNESS_MAX_GAP_DAYS}). As probabilidades do modelo não '
                f'refletem lutas recentes.</div>')
    return (f'<div class="status ok"><span class="dot"></span>Base de dados em dia — '
            f'evento mais recente há {gap} dia(s).</div>')


# ---------------------------------------------------------------------------
# a ficha de uma luta
# ---------------------------------------------------------------------------

def _lado(f: dict, lado: str) -> dict:
    a = lado == "a"
    nome = f["fighter_a"] if a else f["fighter_b"]
    return {
        "nome": nome,
        "modelo": f["model_prob_a"] if a else f["model_prob_b"],
        "mercado": f["market_prob_a"] if a else f["market_prob_b"],
        "odd_casa": (f["odds_a_casa"] if a else f["odds_b_casa"]) if f.get("odds_da_casa") else None,
        "entrada": f.get(f"entrada_{lado}"),
        "do_modelo": f["model_side"] == nome,
    }


def _marca_entrada(entrada) -> str:
    if entrada is None or entrada <= 1:
        return ""
    return (f'<em class="corner-edge" title="A casa paga acima do preço justo da '
            f'Pinnacle sem margem neste lado">sinal {entrada:.3f}</em>')


def _medidas(s: dict, lado: str) -> str:
    return (f'<dl class="med med-{lado}">'
            f'<div><dt>modelo</dt><dd class="v-mod">{_curto(_pct(s["modelo"]))}</dd></div>'
            f'<div><dt>mercado</dt><dd class="v-mer">{_curto(_pct(s["mercado"]))}</dd></div>'
            f'<div><dt>casa</dt><dd>{_curto(_odd(s["odd_casa"]))}{_marca_entrada(s["entrada"])}</dd></div>'
            f'</dl>')


def _delta_lado(f: dict) -> float:
    """modelo − mercado PARA O LADO QUE O MODELO APONTA."""
    if f["model_side"] == f["fighter_a"]:
        return f["model_prob_a"] - f["market_prob_a"]
    return f["model_prob_b"] - f["market_prob_b"]


def _detalhes(f: dict, sa: dict, sb: dict, p: dict) -> str:
    from src.card_report import _METHOD_LABELS, _matched_note

    def linha(rot, va, vb, nota=""):
        return (f'<tr><th scope="row">{rot}{f"<small>{nota}</small>" if nota else ""}</th>'
                f'<td>{va}</td><td>{vb}</td></tr>')

    def entrada(s):
        return INDISPONIVEL if s["entrada"] is None else f'{s["entrada"]:.3f}'

    mp = f.get("method_probs") or {}
    metodo = (" · ".join(f"{rot} {_pct(mp.get(k))}" for k, rot in _METHOD_LABELS)
              if mp else INDISPONIVEL)
    ev = f.get("ev")
    ev_txt = INDISPONIVEL if ev is None else (
        f'{ev:.2f} para {_e(f["model_side"])}'
        + ("" if f.get("odds_da_casa") else
           " (calculado na mediana: não é um preço que dá para pegar)"))
    sharp = (f'{_pct(float(p["sharp_prob"]))} para {_e(f["model_side"])} · lida '
             f'{_quando(p.get("sharp_lido_em"))}' if p.get("sharp_prob") else INDISPONIVEL)
    motivos = []
    if p.get("bloqueio"):
        motivos.append(f'Cotação bloqueada: {_e(p["bloqueio"])}.')
    if p.get("fonte_card") and p["fonte_card"] != "ambas":
        motivos.append(f'Confronto conhecido por uma fonte só ({_e(p["fonte_card"])}).')
    if not f.get("odds_da_casa"):
        motivos.append("Sem odd da casa: não há preço executável registrado.")
    origem = ""
    if p:
        origem = f"""
          <li><b>Versão da previsão</b> <code>{_e(p.get("prediction_id") or INDISPONIVEL)}</code>
            · congelada {_quando(p.get("criado_em"))} · modelo {_e(p.get("model_name") or INDISPONIVEL)}
            · manifesto <code>{_e((p.get("manifesto") or INDISPONIVEL)[:12])}</code></li>
          <li><b>Fontes do confronto</b> {_e(p.get("fonte_card") or INDISPONIVEL)}
            · casamento {_e(p.get("casamento") or INDISPONIVEL)}</li>
          <li><b>Referências</b> casa <code>{_e(p.get("ref_tp_a") or "—")}</code> /
            <code>{_e(p.get("ref_tp_b") or "—")}</code> · Pinnacle
            <code>{_e(p.get("sharp_ref") or "—")}</code></li>"""
    tp_hora = _quando(p.get("tp_lido_em")) if p else INDISPONIVEL
    return f"""
      <details class="det">
        <summary>Detalhes <span class="det-sub">comparação, origem e bloqueios</span></summary>
        <table class="cmp">
          <thead><tr><th></th><th scope="col">{_e(sa["nome"])}</th><th scope="col">{_e(sb["nome"])}</th></tr></thead>
          <tbody>
            {linha("Modelo", _pct(sa["modelo"]), _pct(sb["modelo"]), "probabilidade estimada")}
            {linha("Mercado", _pct(sa["mercado"]), _pct(sb["mercado"]), "mediana das casas, sem margem")}
            {linha("Odd da casa", _odd(sa["odd_casa"]), _odd(sb["odd_casa"]), f"lida {tp_hora}")}
            {linha("Sinal de entrada", entrada(sa), entrada(sb), "Pinnacle sem margem × odd da casa")}
          </tbody>
        </table>
        <ul class="orig">
          <li><b>EV do lado do modelo</b> {ev_txt} — auto-referente: usa a
            probabilidade do próprio modelo. Critério de pré-registro do paper
            trading, não lucro esperado.</li>
          <li><b>Pinnacle no registro</b> {sharp}</li>
          <li><b>Método previsto</b> {metodo}</li>{origem}
        </ul>
        {"".join(f'<p class="bloq">{m}</p>' for m in motivos)}
        {_matched_note(f)}
      </details>"""


def _ficha(f: dict, p: dict, principal: bool, numero: int) -> str:
    from src.card_report import _delta_band, _gap_bar
    from src.prediction_history import avatar_html

    sa, sb = _lado(f, "a"), _lado(f, "b")
    delta = _delta_lado(f)
    estado, _ = _delta_band(delta)
    diverge = f["model_side"] != f["favorite"]
    ev = f.get("ev")
    tags = [f'<span class="tag t-mod">modelo: {_e(f["model_side"])}</span>',
            f'<span class="tag t-mer">favorito do mercado: {_e(f["favorite"])}</span>']
    if ev is not None and ev > 1:
        tags.append('<span class="tag t-crit" title="Critério de registro do paper '
                    'trading. Não é lucro esperado nem vantagem demonstrada.">'
                    'EV&gt;1 pelo critério de registro'
                    + ("" if f.get("odds_da_casa") else " (na mediana)") + '</span>')
    # Bloqueio de DADO ou de IDENTIDADE: sem preco da casa, cotacao sem
    # casamento inequivoco, ou confronto de uma fonte so. O sinal de entrada
    # vencido (preco mais velho que a validade) e estado de TEMPO, vale para
    # o card inteiro e ja aparece nos detalhes e na proveniencia -- contado
    # aqui, marcaria todas as lutas e o filtro nao separaria nada.
    bloq = (not f.get("odds_da_casa") or bool(p.get("bloqueio"))
            or p.get("fonte_card", "ambas") not in ("", "ambas"))
    if bloq:
        tags.append('<span class="tag t-bloq">bloqueio de dado</span>')
    filtros = " ".join(x for x in ("diverge" if diverge else "",
                                   "casa" if f.get("odds_da_casa") else "",
                                   "bloq" if bloq else "") if x)
    inicio = p.get("inicio", "") if p else ""
    rotulo = "Luta principal" if principal else f"Luta {numero}"
    return f"""
    <article class="ficha{' principal' if principal else ''}" data-f="{filtros}" data-state="{estado}">
      <header class="f-top">
        <span class="ordem">{rotulo}</span>
        <span class="inicio">início {_quando(inicio) if inicio else INDISPONIVEL}</span>
      </header>
      <div class="frente">
        <div class="atleta a{' modelo' if sa['do_modelo'] else ''}">{avatar_html(sa["nome"], big=principal)}
          <h3>{_e(sa["nome"])}</h3></div>
        <div class="atleta b{' modelo' if sb['do_modelo'] else ''}">{avatar_html(sb["nome"], big=principal)}
          <h3>{_e(sb["nome"])}</h3></div>
      </div>
      {_gap_bar(f["model_prob_a"], f["market_prob_a"], delta, f["model_side"] == f["fighter_a"])}
      <div class="meds">{_medidas(sa, "a")}{_medidas(sb, "b")}</div>
      <div class="tags">{"".join(tags)}</div>
      {_detalhes(f, sa, sb, p)}
    </article>"""


# ---------------------------------------------------------------------------
# metodologia (texto curto, so o que o projeto ja define)
# ---------------------------------------------------------------------------

METODOLOGIA = """
  <dl class="glos">
    <div><dt>Modelo</dt><dd>Probabilidade de vitória estimada pelo modelo, congelada
      antes da luta. É uma estimativa, não uma medida de confiança: o modelo é bem
      calibrado, mas separa pouco os lutadores, e o mercado o supera (log loss 0.502
      contra 0.622 nas lutas casadas, README).</dd></div>
    <div><dt>Mercado</dt><dd>Mediana das odds de 20+ casas, sem a margem (método
      proporcional). É a referência contra a qual o modelo é medido. Não é um preço
      que dá para pegar.</dd></div>
    <div><dt>Casa</dt><dd>Odd da casa onde a entrada seria feita, lida no
      pré-registro, com horário e referência nos detalhes. Não é atualizada
      depois. Sem ela, o critério de registro cai na mediana.</dd></div>
    <div><dt>Delta</dt><dd>Modelo menos mercado, no lado que o modelo aponta, em
      pontos de probabilidade. O vão vermelho na barra mostra a divergência; não
      quer dizer que a divergência seja boa.</dd></div>
    <div><dt>Favorito, critério EV&gt;1 e vantagem</dt><dd>São três coisas
      diferentes. <b>Favorito</b> é o lado que o mercado considera mais provável.
      <b>EV&gt;1</b> é o critério de pré-registro do paper trading (probabilidade do
      modelo × odd da casa); é auto-referente e não é lucro esperado.
      <b>Vantagem demonstrada</b> não existe hoje.</dd></div>
    <div><dt>Sinal de entrada</dt><dd>Probabilidade da Pinnacle sem margem × odd
      da casa, nos dois lados. Acima de 1, a casa paga mais que o preço justo da
      Pinnacle naquele instante. Não depende do modelo.</dd></div>
    <div><dt>CLV</dt><dd>Probabilidade da Pinnacle no fechamento menos a do
      registro, no lado do modelo. A regra original segue até 200 pernas; o legado
      não tem prova por preço, e nada disso é vantagem demonstrada.</dd></div>
    <div><dt>Simulação</dt><dd>Unidades (u) e P&amp;L do histórico são paper
      trading, 1 unidade por perna, sem aposta real. P&amp;L não é variável de
      decisão.</dd></div>
    <div><dt>Indisponível</dt><dd>Dado não registrado. A página não preenche
      lacunas com dado de hoje: avisos de experiência que a versão não guardou
      aparecem como não registrados.</dd></div>
    <div><dt>Horários</dt><dd>Horário de Brasília. Início das lutas pela casa; leitura
      da mediana não tem horário por luta no card.</dd></div>
  </dl>"""


# ---------------------------------------------------------------------------
# pagina
# ---------------------------------------------------------------------------

def render_pagina(analysis: dict, freshness_gap_days, *, card_name: str = "",
                  history_panel_html: str = "", event_date: str = "",
                  gerado_em: str = "", fingerprint: str = "",
                  model_name: str = "", prov_html: str = "") -> str:
    from src.card_report import _duas_odds_nota, _format_event_date
    from src.design import (FONT_NOTICE, WORDMARK, favicon_data_uri, fonts_css,
                            mark_svg)

    prov = proveniencia_do_evento(event_date)
    lutas = sorted(analysis["favorites"] + analysis["underdogs"],
                   key=lambda f: f.get("card_order", 999))
    principal = analysis.get("main_event") or {}
    par_principal = (principal.get("fighter_a"), principal.get("fighter_b"))
    fichas, n = [], 0
    for f in lutas:
        eh = (f["fighter_a"], f["fighter_b"]) == par_principal
        n += 0 if eh else 1
        fichas.append(_ficha(f, prov.get(frozenset((f["fighter_a"], f["fighter_b"])), {}),
                             eh, n + 1))
    sem = analysis.get("no_prediction") or []
    sem_html = ""
    if sem:
        itens = "".join(f'<li><b>{_e(f["fighter_a"])}</b> vs <b>{_e(f["fighter_b"])}</b>'
                        f' — {_e(f["reason"])}</li>' for f in sem)
        sem_html = (f'<section class="sem"><h2>Sem previsão ({len(sem)})</h2>'
                    f'<p>O modelo não tem como estimar estas lutas; elas ficam fora '
                    f'dos filtros acima.</p><ul>{itens}</ul></section>')
    n_casa = sum(1 for f in lutas if f.get("odds_da_casa"))
    criados = sorted(p["criado_em"] for p in prov.values() if p.get("criado_em"))
    inicios = sorted(p["inicio"] for p in prov.values() if p.get("inicio"))
    janela = f"{_quando(inicios[0])} – {_quando(inicios[-1])}" if inicios else INDISPONIVEL
    data_txt = _format_event_date(event_date) if event_date else INDISPONIVEL
    titulo = card_name or "Card analisado"
    carimbo = (f'<meta name="fm-generator" content="{fingerprint}">' if fingerprint else "")
    historico = history_panel_html or (
        '<p class="nota">Nenhum evento registrado ainda.</p>')
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
{carimbo}
<title>{_e(titulo)} — {WORDMARK}</title>
<link rel="icon" href="{favicon_data_uri()}">
<style>
{fonts_css()}
:root {{
  --bg:#0E1012; --panel:#15181C; --panel2:#1B1F24; --line:#262A30;
  --text:#EEECE7; --dim:#B9BCC2; --muted:#8C929B;
  --mod:#D6AF37; --mer:#8FAEE0; --casa:#D20A11; --casa-txt:#F2615C;
  --ok:#3FB37A; --alerta:#E08A5B;
  --disp:"FM Display","Archivo Narrow","Arial Narrow",sans-serif;
  --sans:"FM Sans",-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
}}
* {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; scroll-behavior:smooth; scroll-padding-top:64px; }}
@media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 var(--sans);
  font-variant-numeric:tabular-nums; }}
a {{ color:inherit; }}
:focus-visible {{ outline:2px solid var(--mod); outline-offset:2px; border-radius:3px; }}
h2 {{ font:800 italic 1.6rem/1 var(--disp); text-transform:uppercase; margin:0 0 6px; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:0 20px; }}
.topo {{ position:sticky; top:0; z-index:5; background:rgba(14,16,18,.94); border-bottom:1px solid var(--line);
  backdrop-filter:blur(6px); }}
.topo .wrap {{ display:flex; align-items:center; gap:10px; min-height:52px; }}
.topo .mark {{ width:22px; height:22px; flex:none; }}
.marca {{ font:800 italic 1.05rem/1 var(--disp); letter-spacing:.02em; text-transform:uppercase; }}
.topo nav {{ margin-left:auto; display:flex; gap:4px; }}
.topo nav a {{ text-decoration:none; color:var(--muted); padding:6px 10px; border-radius:6px; font-size:.85rem; }}
.topo nav a[aria-current="true"] {{ color:var(--text); background:var(--panel2); }}
.evento {{ padding:36px 0 22px; }}
.evento .kick {{ color:var(--casa-txt); font:600 .78rem/1 var(--sans); letter-spacing:.14em; text-transform:uppercase; }}
.evento h1 {{ margin:10px 0 6px; font:800 italic clamp(2.2rem,6vw,4.2rem)/.95 var(--disp);
  text-transform:uppercase; }}
.evento .sub {{ color:var(--dim); margin:0; }}
.fatos {{ display:flex; flex-wrap:wrap; gap:6px 22px; margin:18px 0 0; padding:0; list-style:none;
  color:var(--muted); font-size:.85rem; }}
.fatos b {{ color:var(--text); font-weight:600; }}
.status {{ margin-top:14px; font-size:.85rem; color:var(--dim); display:flex; gap:8px; align-items:baseline; }}
.status .dot {{ flex:none; width:8px; height:8px; border-radius:50%; background:var(--muted); transform:translateY(-1px); }}
.status.ok .dot {{ background:var(--ok); }}
.status.bad {{ color:var(--alerta); }} .status.bad .dot {{ background:var(--alerta); }}
.aviso {{ color:var(--muted); font-size:.82rem; margin:8px 0 0; }}
.legenda {{ display:flex; flex-wrap:wrap; gap:6px 18px; align-items:center; margin:8px 0 6px;
  color:var(--muted); font-size:.8rem; }}
.legenda i {{ display:inline-block; width:14px; height:6px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.tab-explain {{ color:var(--muted); font-size:.82rem; margin:6px 0 10px; max-width:75ch; }}
.tab-explain strong {{ color:var(--dim); }}
.filtros {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 18px; }}
.filtros button {{ font:600 .82rem/1 var(--sans); color:var(--dim); background:transparent;
  border:1px solid var(--line); border-radius:999px; padding:8px 12px; cursor:pointer; }}
.filtros button[aria-pressed="true"] {{ color:var(--bg); background:var(--text); border-color:var(--text); }}
.ficha {{ padding:22px 0 18px; border-top:1px solid var(--line); }}
.ficha.principal {{ border-top:2px solid var(--casa); }}
.f-top {{ display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem;
  letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px; }}
.principal .ordem {{ color:var(--casa-txt); font-weight:600; }}
.frente {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:center; }}
.atleta {{ display:flex; align-items:center; gap:12px; min-width:0; }}
.atleta.b {{ flex-direction:row-reverse; text-align:right; }}
.atleta h3 {{ margin:0; font:800 italic clamp(1.25rem,3.4vw,2.1rem)/1 var(--disp);
  text-transform:uppercase; color:var(--dim); overflow-wrap:anywhere; }}
.atleta.modelo h3 {{ color:var(--text); }}
.principal .atleta h3 {{ font-size:clamp(1.6rem,5vw,3rem); }}
.avatar {{ flex:none; position:relative; width:52px; height:52px; border-radius:50%; overflow:hidden;
  background:var(--mono,var(--panel2)); display:inline-grid; place-items:center; vertical-align:middle; }}
.avatar.lg {{ width:72px; height:72px; }}
.avatar.sm {{ width:24px; height:24px; }}
.avatar-txt {{ color:var(--dim); font:600 .85rem var(--sans); }}
.avatar.sm .avatar-txt {{ font-size:.6rem; }}
.avatar img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:top; }}
.atleta.modelo .avatar {{ box-shadow:0 0 0 2px var(--mod); }}
.gap {{ display:block; margin:16px 0 10px; }}
.gap-num, .gap-legend span:not(.gap-delta) {{ display:none; }}
.gap-track {{ position:relative; display:flex; height:10px; background:var(--panel2); border-radius:5px; }}
.seg {{ height:100%; }}
.seg.l {{ background:var(--mod); border-radius:5px 0 0 5px; }}
.seg.r {{ background:transparent; }}
.gap-span {{ position:absolute; top:50%; transform:translateY(-50%); background:var(--casa-txt); opacity:.85; }}
.gap-mark {{ position:absolute; top:-5px; bottom:-5px; width:3px; margin-left:-1.5px; background:var(--mer); border-radius:2px; }}
.gap-legend {{ display:flex; justify-content:center; margin:-4px 0 6px; }}
.gap-delta {{ color:var(--muted); font-size:.78rem; }}
.ficha[data-state="material"] .gap-delta, .ficha[data-state="large"] .gap-delta {{ color:var(--casa-txt); }}
.gap-delta small {{ font-size:inherit; margin-left:1px; }}
.meds {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.med {{ display:flex; gap:18px; margin:0; }}
.med-b {{ justify-content:flex-end; text-align:right; }}
.med div {{ min-width:0; }}
.med dt {{ color:var(--muted); font-size:.75rem; }}
.med dd {{ margin:2px 0 0; font-weight:600; font-size:1.02rem; }}
.v-mod {{ color:var(--mod); }} .v-mer {{ color:var(--mer); }}
.corner-edge {{ display:block; font-style:normal; font-weight:600; font-size:.75rem; color:var(--ok); }}
.tags {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }}
.tag {{ font-size:.76rem; padding:3px 9px; border-radius:999px; border:1px solid var(--line); color:var(--dim); }}
.t-mod {{ border-color:rgba(214,175,55,.45); }}
.t-mer {{ border-color:rgba(143,174,224,.45); }}
.t-crit {{ border-style:dashed; }}
.t-bloq {{ color:var(--alerta); border-color:rgba(224,138,91,.5); }}
.det {{ margin-top:12px; }}
.det summary {{ cursor:pointer; color:var(--dim); font-size:.85rem; padding:6px 0; list-style:none; }}
.det summary::-webkit-details-marker {{ display:none; }}
.det summary::before {{ content:"+"; display:inline-block; width:1.1em; color:var(--muted); }}
.det[open] summary::before {{ content:"–"; }}
.det-sub {{ color:var(--muted); margin-left:6px; }}
.cmp {{ width:100%; border-collapse:collapse; margin:8px 0 10px; font-size:.86rem; table-layout:fixed; }}
.cmp th, .cmp td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; overflow-wrap:anywhere; }}
.cmp thead th {{ color:var(--dim); font-weight:600; }}
.cmp th[scope="row"] {{ text-align:left; font-weight:600; color:var(--text); }}
.cmp small {{ display:block; color:var(--muted); font-weight:400; font-size:.8rem; }}
.orig {{ margin:0; padding:0; list-style:none; color:var(--dim); font-size:.84rem; }}
.orig li {{ padding:4px 0; overflow-wrap:anywhere; }}
.orig b {{ color:var(--text); font-weight:600; margin-right:6px; }}
code {{ font:.78rem ui-monospace,Consolas,monospace; color:var(--dim); word-break:break-all; }}
.bloq {{ color:var(--alerta); font-size:.84rem; margin:8px 0 0; }}
.note {{ color:var(--muted); font-size:.82rem; margin:8px 0 0; }}
.nd {{ color:var(--muted); font-weight:400; text-decoration:none; cursor:help; }}
.sr {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
.sem {{ border-top:1px solid var(--line); padding:22px 0; color:var(--dim); font-size:.9rem; }}
.sem h2 {{ font-size:1.2rem; }} .sem ul {{ padding-left:18px; }}
.prov {{ margin:4px 0 14px; font-size:.84rem; color:var(--dim); }}
.prov summary {{ cursor:pointer; color:var(--dim); padding:6px 0; }}
.prov-corpo {{ border-left:2px solid var(--line); padding:4px 0 4px 14px; }}
.prov-corpo h3 {{ font-size:.85rem; margin:12px 0 4px; }}
.prov-lista {{ margin:0; padding-left:18px; }}
.secao {{ border-top:2px solid var(--line); margin-top:30px; padding-top:28px; }}
.secao > p {{ color:var(--dim); max-width:75ch; margin:0 0 18px; }}
/* historico: classes de prediction_history.render_history_panel */
.clv-alert {{ border-left:3px solid var(--alerta); padding:10px 14px; margin:0 0 18px; background:var(--panel); }}
.clv-alert-title {{ color:var(--alerta); font-weight:600; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; margin-bottom:4px; }}
.clv-alert-date {{ color:var(--muted); font-size:.8rem; }}
.clv-cmd {{ display:block; margin-top:6px; }}
.serie-box {{ margin:0 0 22px; }}
.serie-title {{ font-weight:600; margin-bottom:10px; }}
.serie-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px 22px; }}
.serie-stat {{ display:flex; flex-direction:column; border-top:1px solid var(--line); padding-top:8px; }}
.serie-label {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
.serie-val {{ font-size:1.5rem; font-weight:600; }}
.serie-val.pos {{ color:var(--ok); }} .serie-val.neg {{ color:var(--alerta); }}
.serie-sub {{ color:var(--muted); font-size:.8rem; }}
.hist-date {{ color:var(--muted); font-weight:400; font-size:.8rem; margin-left:6px; }}
.hist-event {{ border-top:1px solid var(--line); padding:14px 0; }}
.hist-head {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; align-items:center; }}
.hist-toggle {{ all:unset; cursor:pointer; display:flex; flex-wrap:wrap; justify-content:space-between;
  gap:8px; align-items:center; width:100%; min-height:44px; }}
.hist-toggle::before {{ content:"+"; color:var(--muted); width:1em; }}
.hist-toggle[aria-expanded="true"]::before {{ content:"–"; }}
.hist-title {{ font:800 italic 1.15rem/1.1 var(--disp); text-transform:uppercase; flex:1; }}
.hist-score, .hist-badge {{ font-size:.78rem; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--dim); margin-left:4px; white-space:nowrap; }}
.hist-score.pos {{ color:var(--ok); }} .hist-score.neg {{ color:var(--alerta); }}
.hist-badge.hit {{ color:var(--ok); border-color:rgba(63,179,122,.45); }}
.hist-badge.miss {{ color:var(--alerta); border-color:rgba(224,138,91,.45); }}
.hist-scroll {{ margin-top:10px; }}
.hist-table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.hist-table th {{ text-align:left; color:var(--muted); font-weight:600; font-size:.75rem; padding:6px 8px; border-bottom:1px solid var(--line); }}
.hist-table td {{ padding:8px; border-bottom:1px solid var(--line); vertical-align:middle; }}
.hist-prob {{ color:var(--mod); font-weight:600; }}
.hist-pending {{ color:var(--muted); }}
.vs {{ color:var(--muted); margin:0 4px; }}
.glos {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:4px 28px; margin:0; }}
.glos div {{ border-top:1px solid var(--line); padding:12px 0; }}
.glos dt {{ font-weight:600; margin-bottom:4px; }}
.glos dd {{ margin:0; color:var(--dim); font-size:.88rem; }}
footer {{ border-top:1px solid var(--line); margin-top:30px; padding:20px 0 40px; color:var(--muted); font-size:.75rem; }}
@media (max-width:640px) {{
  .wrap {{ padding:0 16px; }}
  .marca {{ display:none; }}
  .topo nav a, .filtros button {{ min-height:44px; display:inline-flex; align-items:center; }}
  .topo nav a {{ padding:6px 8px; font-size:.82rem; }}
  .det summary, .prov summary {{ min-height:44px; display:flex; align-items:center; }}
  .frente {{ gap:10px; }}
  .atleta {{ flex-direction:column; align-items:flex-start; gap:8px; }}
  .atleta.b {{ flex-direction:column; align-items:flex-end; }}
  .avatar {{ width:44px; height:44px; }} .avatar.lg {{ width:56px; height:56px; }}
  .avatar.sm {{ width:22px; height:22px; }}
  .meds {{ gap:10px; }} .med {{ gap:10px; }}
  .med dd {{ font-size:.9rem; }}
  .cmp th, .cmp td {{ padding:6px 4px; }}
  .hist-table thead {{ display:none; }}
  .hist-table, .hist-table tbody, .hist-table tr, .hist-table td {{ display:block; width:100%; }}
  .hist-table tr {{ border-bottom:1px solid var(--line); padding:8px 0; }}
  .hist-table td {{ border:0; padding:3px 0; }}
  .hist-table td:nth-child(2)::before {{ content:"modelo: "; color:var(--muted); }}
  .hist-table td:nth-child(3)::before {{ content:"mercado: "; color:var(--muted); }}
  .hist-table td:nth-child(4)::before {{ content:"vencedor: "; color:var(--muted); }}
}}
@media (prefers-reduced-motion:no-preference) {{
  .seg.l {{ transition:width .5s ease; }}
}}
</style>
</head>
<body>
<header class="topo"><div class="wrap">{mark_svg()}<span class="marca">{WORDMARK}</span>
  <nav aria-label="Seções"><a href="#evento" aria-current="true">Evento</a><a href="#historico">Histórico</a><a href="#metodologia">Metodologia</a></nav>
</div></header>
<main class="wrap">
  <section class="evento" id="evento" aria-labelledby="ev-titulo">
    <div class="kick">Pré-registro · {_e(data_txt)}</div>
    <h1 id="ev-titulo">{_e(titulo)}</h1>
    <p class="sub">Probabilidade do modelo contra a referência do mercado, luta a luta.</p>
    <ul class="fatos">
      <li><b>{len(lutas)}</b> lutas com previsão</li>
      <li><b>{n_casa}</b> com odd da casa</li>
      <li>previsões congeladas <b>{_quando(criados[-1]) if criados else INDISPONIVEL}</b></li>
      <li>início das lutas <b>{janela}</b> (Brasília)</li>
      <li>modelo <b>{_e(model_name or analysis.get("model_name") or INDISPONIVEL)}</b></li>
    </ul>
    {frescor_html(freshness_gap_days)}
    <p class="aviso">Estimativa estatística, não é recomendação de aposta. Página gerada {_e(gerado_em) or INDISPONIVEL}.</p>
  </section>

  <section id="lutas" aria-label="Lutas">
    <div class="legenda"><span><i style="background:var(--mod)"></i>modelo</span>
      <span><i style="background:var(--mer);width:3px;height:12px"></i>mercado</span>
      <span><i style="background:var(--casa-txt);height:3px"></i>divergência</span>
      {'<span>casa = odd da Thunderpick</span>' if n_casa else ''}</div>
    {_duas_odds_nota(analysis)}
    {prov_html}
    <div class="filtros" role="group" aria-label="Filtrar lutas">
      <button type="button" data-f="" aria-pressed="true">Todas</button>
      <button type="button" data-f="diverge" aria-pressed="false">Modelo contra o favorito</button>
      <button type="button" data-f="casa" aria-pressed="false">Com odd da casa</button>
      <button type="button" data-f="bloq" aria-pressed="false">Bloqueio de dado</button>
    </div>
    <p class="vazio note" hidden>Nenhuma luta neste filtro.</p>
    {"".join(fichas)}
    {sem_html}
  </section>

  <section class="secao" id="historico" aria-labelledby="h-hist">
    <h2 id="h-hist">Histórico</h2>
    <p>Previsões congeladas antes de cada luta e o que aconteceu. Unidades (u) são
    simulação de paper trading — 1 unidade por perna EV&gt;1, sem aposta real — e
    não decidem nada. Cada placar traz o número de lutas em que foi medido.</p>
    {historico}
  </section>

  <section class="secao" id="metodologia" aria-labelledby="h-met">
    <h2 id="h-met">Metodologia</h2>
    <p>O que cada número quer dizer, e o que ele não quer dizer.</p>
    {METODOLOGIA}
  </section>
</main>
<footer><div class="wrap">Estimativa estatística, não é recomendação de aposta. Nenhuma vantagem sobre o mercado está demonstrada. {FONT_NOTICE}</div></footer>
<script>
(function(){{
  var bs=document.querySelectorAll('.filtros button'), fs=document.querySelectorAll('.ficha'),
      vazio=document.querySelector('.vazio');
  bs.forEach(function(b){{ b.addEventListener('click', function(){{
    bs.forEach(function(x){{ x.setAttribute('aria-pressed', x===b?'true':'false'); }});
    var k=b.getAttribute('data-f'), n=0;
    fs.forEach(function(f){{ var esc=!!k && (' '+f.getAttribute('data-f')+' ').indexOf(' '+k+' ')<0;
      f.hidden=esc; if(!esc) n++; }});
    if (vazio) vazio.hidden = n>0;
  }}); }});
  document.querySelectorAll('.hist-event').forEach(function(ev, i){{
    var head=ev.querySelector('.hist-head'), corpo=ev.querySelector('.hist-scroll');
    if(!head||!corpo) return;
    var b=document.createElement('button'); b.type='button'; b.className='hist-toggle';
    corpo.id='hist-'+i; b.setAttribute('aria-controls', corpo.id);
    while(head.firstChild) b.appendChild(head.firstChild);
    head.appendChild(b);
    var aberto = i<1; b.setAttribute('aria-expanded', aberto); corpo.hidden=!aberto;
    b.addEventListener('click', function(){{ var a=b.getAttribute('aria-expanded')!=='true';
      b.setAttribute('aria-expanded', a); corpo.hidden=!a; }});
  }});
  var links=document.querySelectorAll('.topo nav a');
  if ('IntersectionObserver' in window) {{
    var io=new IntersectionObserver(function(es){{ es.forEach(function(e){{ if(e.isIntersecting){{
      var alvo = e.target.id==='lutas' ? 'evento' : e.target.id;
      links.forEach(function(l){{ l.setAttribute('aria-current', l.getAttribute('href')==='#'+alvo?'true':'false'); }});
    }} }}); }}, {{rootMargin:'-45% 0px -50% 0px'}});
    ['evento','lutas','historico','metodologia'].forEach(function(id){{ var s=document.getElementById(id); if(s) io.observe(s); }});
  }}
}})();
</script>
</body>
</html>"""
