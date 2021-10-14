SELECT
    cut_id
FROM
    dna_ml.cluster_metadata
WHERE
   name = {{ name }} AND
   run_date = {{ run_date }} AND
   cluster_size = {{ cluster_size }} AND
   features = {{ features }} AND
   hyperparameters = {{ hyperparameters }} AND
   results = {{ results }} AND
   notes = {{ notes }} AND
   insert_date = {{ insert_date }}