#!/usr/bin/env bash
set -e

NER_YML="config/parameters/ner.yml"

PROMPTS=(
  "method_1_var"
  "method_2_context"
  "method_3_guides_inj"
  "method_4_full_gen"
)

echo "========================================="
echo "Running NER experiments for all prompts"
echo "Config file: ${NER_YML}"
echo "========================================="

for PROMPT in "${PROMPTS[@]}"; do
  echo
  echo "-----------------------------------------"
  echo "Setting ner.prompt = ${PROMPT}"
  echo "-----------------------------------------"

  yq -i ".ner.prompt = \"${PROMPT}\"" "${NER_YML}"

  echo "Current ner.prompt:"
  yq ".ner.prompt" "${NER_YML}"

  echo "Launching training..."
  python run.py

  echo "Finished run for prompt: ${PROMPT}"
done

echo
echo "========================================="
echo "All experiments completed successfully"
echo "========================================="
