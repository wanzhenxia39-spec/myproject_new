# PAMAP2 GZSL split

This directory reuses the 20 PAMAP2 seen/unseen class splits from
`split/zero_shot_split` and changes only the evaluation protocol:

- training classes: the 8 `seen_activities` of each split;
- evaluation samples: 50 held-out windows for every seen and unseen activity
  (12 x 50 = 600 windows per split);
- classification candidates: `seen_activities + unseen_activities`;
- leakage prevention: the 400 held-out seen windows are removed before the
  attribute-model train/validation split;
- primary metrics: seen accuracy, unseen accuracy, and their harmonic mean.

Run one split with the PAMAP2 pipeline (use the A2 vector JSON belonging to the
same split):

```bash
python pamap2_pipeline/pamap_pipeline.py \
  --evaluation_protocol gzsl \
  --unseen_count 4 \
  --only_split_index 13 \
  --split_manifest_json /home/wanzhen/split/gzsl/pamap2_unseen4_random20_gzsl_splits_seed42.json \
  --sample_manifest_json /home/wanzhen/split/gzsl/pamap2_unseen4_random20_gzsl_eval50_samples_seed42.json \
  --samples_per_unseen_activity 50 \
  --features /home/wanzhen/windows/pamap_windows/all_features_20_120.npy \
  --feature_names /home/wanzhen/windows/pamap_windows/all_features_name_20_120.npy \
  --labels /home/wanzhen/windows/pamap_windows/label_20_120.npy \
  --attributes_json PATH/TO/split_13/activity_attribute_vectors_a2.json \
  --model_root PATH/TO/gzsl_models \
  --results_csv PATH/TO/gzsl_results.csv \
  --detailed_dir PATH/TO/gzsl_details
```

The legacy argument name `--samples_per_unseen_activity` is retained for
backward compatibility; under `--evaluation_protocol gzsl` it applies to all
12 evaluation activities.
