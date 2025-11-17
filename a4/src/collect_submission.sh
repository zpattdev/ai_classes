#! /bin/bash

rm -f assignment4_submission.zip 
pushd submission
echo "Collecting submission files..."

zip -r ../assignment4_submission.zip \
  __init__.py \
  model_embeddings.py \
  nmt_model.py \
  utils.py \
  test_outputs.txt

popd

echo "Done!"