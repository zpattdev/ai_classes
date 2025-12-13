#! /bin/bash

rm -f assignment5_submission.zip 
pushd submission
echo "Collecting submission files..."

zip -r ../assignment5_submission.zip \
  __init__.py \
  attention.py \
  dataset.py \
  helper.py \
  model.py \
  trainer.py \
  utils.py \
  vanilla.model.params \
  vanilla.nopretrain.dev.predictions \
  vanilla.nopretrain.test.predictions \
  vanilla.pretrain.params \
  vanilla.finetune.params \
  vanilla.pretrain.dev.predictions \
  vanilla.pretrain.test.predictions \
  rope.pretrain.params \
  rope.finetune.params \
  rope.pretrain.dev.predictions \
  rope.pretrain.test.predictions

popd

echo "Done!"