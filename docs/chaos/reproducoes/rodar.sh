#!/bin/bash
# Reproducoes CHAOS do UFC Predictor, numa COPIA DESCARTAVEL.
#
#   docs/chaos/reproducoes/rodar.sh                 # todos os cenarios
#   docs/chaos/reproducoes/rodar.sh c4_concorrencia # um cenario
#   CHAOS_FONTE=/caminho/da/worktree rodar.sh       # testar uma correcao
#
# Copia CHAOS_FONTE (padrao: este repositorio) sem .git para um diretorio
# temporario (repositorio git vazio, sem remoto), roda sem variaveis de ambiente (sem credenciais)
# e com rede bloqueada. Nada e escrito no repositorio de origem.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
FONTE=${CHAOS_FONTE:-$(cd "$AQUI/../../.." && pwd)}
BASE=$(mktemp -d "${TMPDIR:-/tmp}/ufc-chaos.XXXXXX")
# copia SEM .git: numa worktree o .git e um ponteiro para o repositorio
# principal, e qualquer comando git na copia alteraria o repositorio real.
mkdir -p "$BASE/app"
tar -C "$FONTE" --exclude=./.git -cf - . | tar -C "$BASE/app" -xf -
git init -q "$BASE/app"
mkdir -p "$BASE/limpo" "$BASE/guard" "$BASE/home"
cp -a "$BASE/app/data" "$BASE/app/docs" "$BASE/limpo/"
cp "$AQUI/bloqueio_rede.py" "$BASE/guard/sitecustomize.py"
cat > "$BASE/py.sh" <<FIM
#!/bin/bash
cd "$BASE/app"
exec env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$BASE/home" \
  PYTHONPATH="$BASE/guard:$AQUI:$BASE/app" CHAOS_COPIA="$BASE/app" \
  CHAOS_PY="$BASE/py.sh" TZ=UTC LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 python "\$@"
FIM
chmod +x "$BASE/py.sh"
CENARIOS=${*:-"c1_api c2_interrupcao c3_truncado c4_concorrencia c5_c6_fechamento_confronto c7_tempo c8_ui"}
echo "copia descartavel: $BASE  (fonte: $(git -C "$FONTE" rev-parse --short HEAD))"
PASSOU=0; DEFEITO=0; ERRO=0
for c in $CENARIOS; do
  echo "################ $c"
  SAIDA="$BASE/saida.$c.txt"
  "$BASE/py.sh" "$AQUI/$c.py" >"$SAIDA" 2>&1
  RC=$?
  # nada e filtrado: so as linhas de log [INFO]/[WARNING] de bibliotecas
  # saem da exibicao; [DEFEITO], [ERRO] e tracebacks aparecem sempre.
  grep -v " \[INFO\] \| \[WARNING\] \| \[ERROR\] " "$SAIDA"
  P=$(grep -c "^\[PASSOU\]" "$SAIDA"); D=$(grep -c "^\[DEFEITO\]" "$SAIDA")
  PASSOU=$((PASSOU+P)); DEFEITO=$((DEFEITO+D))
  if [ "$RC" -ne 0 ]; then
    ERRO=$((ERRO+1))
    echo "[ERRO] cenario $c terminou com codigo $RC; saida completa (inclui log):"
    cat "$SAIDA"
  fi
done
echo "================ PASSOU $PASSOU | DEFEITO $DEFEITO | ERRO $ERRO"
echo "(copia descartavel mantida em $BASE)"
[ "$DEFEITO" -eq 0 ] && [ "$ERRO" -eq 0 ] || exit 1
