#!/bin/bash
# Demo-GUI export of a session model = data/lcs/realfirst/cmp/v12_152/post152.sh with every output under the session:
# decoder, head split (beside the run, <s>/train/<tag>_h{1,2}), per-head model jsons, prep148_s1/_s2 (--out, --prep-dir),
# then export_arms.py (configs148 + make_wp x0.5 clone) -> <R>/arm.json. usage: post_export.sh <session> <tag> [<name> = <tag>]
set -u
M=/home/hienbui/git/magna_belt_simulation-main; C=data/lcs/realfirst/cmp; R8=$M/$C/v12_148
L=/home/hienbui/git/lcs_learning
s=$(realpath "$1"); tag=$2; n=${3:-$2}
O=$s/train; R=$O/$tag/export; S=$R/encoder_study; ST=$R/status; G=$R/logs
mkdir -p "$G"
cd $M
say() { echo "$(date -Is) $*" | tee -a "$ST"; }
die() { say "FAIL export $n: $*"; exit 1; }
export LCS_CACHE_REPRO_TOL_REL=1e-5
ck=$(python3 -c "import json; print(json.load(open('$O/$tag/refit_report.json'))['checkpoint'])") || die "no ckpt"
mkdir -p "$S/$n/decoder_decfit"
(cd $L && .venv/bin/python scripts/export_learned_lcs_deploy.py --checkpoint "$ck" --decoder-out "$S/$n/decoder_decfit/decoder.npz") > "$G/dec.log" 2>&1 || die "decoder"
(cd $L && .venv/bin/python $M/$C/v12_129/split_heads.py "$O/$tag") > "$G/split.log" 2>&1 || die "split"
for k in 1 2; do python3 -c "
import json; json.dump({'$n': {'letter': '${n: -1}'.lower(), 'run': '$O/${tag}_h$k', 'decoder': '$S/$n/decoder_decfit/decoder.npz', 'def': 'cmd_delta',
  'about': 'demo GUI $n = $tag (single shared-encoder gated model), stage-$k head'}}, open('$R/${n}_s$k.json', 'w'), indent=1)"; done
uv run --frozen python $R8/prep148_s1.py --models-json "$R/${n}_s1.json" --prefix V39 --out "$S" "$n" > "$G/prep_s1.log" 2>&1 || die "prep s1 $(tail -1 "$G/prep_s1.log")"
uv run --frozen python $R8/prep148_s2.py --models-json "$R/${n}_s2.json" --prefix V39 --out "$S" "$n" --prep-dir "$R/prep_s2" --chains $R8/demos > "$G/prep_s2.log" 2>&1 || die "prep s2 $(tail -1 "$G/prep_s2.log")"
uv run --frozen python apps/demo_gui/backend/export_arms.py "$R" "$S" "$n" > "$G/arms.log" 2>&1 || die "arms $(tail -1 "$G/arms.log")"
say "export $n ($tag) ok: $R/arm.json"
