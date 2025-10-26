#! /bin/bash

rm -f assignment3_submission.zip 
pushd submission
echo "Collecting submission files..."

zip -r ../assignment3_submission.zip \
  __init__.py \
  parser_model.py \
  parser_transitions.py \
  parser_utils.py \
  train.py

popd

echo "Done!"