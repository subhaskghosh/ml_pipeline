INSERT INTO dna_ml.cluster_metadata
        (name, run_date, cluster_size, features, hyperparameters, results, notes, insert_date)
        VALUES
        ({{ name }},
         {{ run_date }},
         {{ cluster_size }},
         {{ features }},
         {{ hyperparameters }},
         {{ results }},
         {{ notes }},
         {{ insert_date }});