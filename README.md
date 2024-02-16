# SQLite-backed Database for Apache TVM's MetaSchedule

This repo contains SQLite-backed Database implementation for [Apache TVM's MetaSchedule](https://discuss.tvm.apache.org/t/rfc-meta-schedule-autotensorir/10120) and its example script. Only tested with TVM v0.15.0 but should work with other versions.

## Directory structure

```
├── LICENSE
├── README.md
├── convert_db.py
├── example_compile_and_run.py
├── example_tune.py
├── log-resnet18
│   ├── database_tuning_record.sqlite
│   ├── database_workload.sqlite
│   └── logs/
└── sql_database.py
```

## Usage

You can use SQLiteDatabase just like JSONDatabase.
All you need to use is to pass SQLiteDatabase instance to the tuning function.

Tuning with JSONDatabase.

```python
tuned_db = ms.relay_integration.tune_relay(mod, params, target, work_dir,
                                           database="json",
                                           max_trials_global=max_trials_global)
```

Tuning with SQLiteDatabase.

```python
sqldb = SQLiteDatabase(work_dir=work_dir)
tuned_db = ms.relay_integration.tune_relay(mod, params, target, work_dir,
                                        database=sqldb,
                                        max_trials_global=max_trials_global)
```

You can also convert existing JSONDatabase with `convert_db.py`.

```bash
$ python convert_db.py --help
usage: dbconvert [-h] [--work_dir WORK_DIR]

options:
  -h, --help           show this help message and exit
  --work_dir WORK_DIR
```

## Example

- `example_tune.py`
  - tunes resnet18 pytorch model and store tuning record to `log-resnet18`. 
- `example_compile_and_run.py`
  - compiles resnet18 pytorch model with the tuning record.
