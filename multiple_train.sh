#!/usr/bin/env bash
set -e

NER_YML="config/parameters/ner.yml"

LLMS=(
  "llama32-3b"
  "gpt-41-nano"
)

PROMPTS=(
  "method_1_var"
  "method_2_guides_inj"
  "method_3_context"
  "method_4_full_gen"
)

echo "========================================="
echo "Running NER experiments"
echo "Config file: ${NER_YML}"
echo "========================================="

# -------------------------------------------------
# BASELINE (NO AUGMENTATION)
# -------------------------------------------------
echo
echo "#########################################"
echo "Running baseline (no augmentation)"
echo "#########################################"

python - <<EOF
import yaml

path = "${NER_YML}"

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["ner"]["data"]["use_augmented"] = False

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

echo "Launching training (no-augmented)..."
python run.py

# -------------------------------------------------
# AUGMENTED RUNS
# -------------------------------------------------
for LLM in "${LLMS[@]}"; do
  echo
  echo "#########################################"
  echo "LLM MODEL: ${LLM}"
  echo "#########################################"

  for PROMPT in "${PROMPTS[@]}"; do
    echo
    echo "-----------------------------------------"
    echo "PROMPT: ${PROMPT} | LLM: ${LLM}"
    echo "-----------------------------------------"

    python - <<EOF
import yaml

path = "${NER_YML}"

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["ner"]["prompt"] = "${PROMPT}"
cfg["ner"]["llm_model"] = "${LLM}"
cfg["ner"]["data"]["use_augmented"] = True

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

    echo "Launching training (augmented)..."
    python run.py

    echo "Finished run: PROMPT=${PROMPT} | LLM=${LLM}"
  done
done


echo
echo "========================================="
echo "All experiments completed successfully"
echo "========================================="
