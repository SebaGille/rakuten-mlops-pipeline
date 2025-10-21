Rakuten MLOps project for Datascientest



Questions:

pourquoi hebergement necessaire pour artifact pour ce projet alors que pas necessaire pour le cours ? a moins que le stockage etait present grace a la VM de l'ecole?

image docker > a publier ? pourquoi si lourde ?


🧪 View experiment at: http://127.0.0.1:5000/#/experiments/1
Traceback (most recent call last):
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/src/models/train_model.py", line 124, in <module>
    main()
    ~~~~^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/src/models/train_model.py", line 115, in main
    mlflow.log_artifact(str(MODEL_FILE))   # baseline_model.pkl (model + vectorizer)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/tracking/fluent.py", line 1445, in log_artifact
    MlflowClient().log_artifact(run_id, local_path, artifact_path)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/tracking/client.py", line 2533, in log_artifact
    self._tracking_client.log_artifact(run_id, local_path, artifact_path)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/tracking/_tracking_service/client.py", line 675, in log_artifact
    artifact_repo.log_artifact(local_path, artifact_path)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/store/artifact/local_artifact_repo.py", line 46, in log_artifact
    mkdir(artifact_dir)
    ~~~~~^^^^^^^^^^^^^^
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/utils/file_utils.py", line 209, in mkdir
    raise e
  File "/Users/sebastien/Documents/DataScientest/sep25_cmlops_rakuten/.venv/lib/python3.13/site-packages/mlflow/utils/file_utils.py", line 206, in mkdir
    os.makedirs(target, exist_ok=True)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen os>", line 218, in makedirs
  File "<frozen os>", line 218, in makedirs
  File "<frozen os>", line 218, in makedirs
  [Previous line repeated 1 more time]
  File "<frozen os>", line 228, in makedirs
OSError: [Errno 45] Operation not supported: '/home/mlflow'