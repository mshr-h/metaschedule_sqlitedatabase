# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sqlite3
import json


def convert_to_db(connect, workload_json, tuning_record_json, output_workload, output_tuning_record):
  for path_db, table_name, sql in [(output_workload, 'workload', 'shash text, mod text'), (output_tuning_record, 'tuning_record', 'workload_index int, trace text, run_secs text, target text, args_info text')]:
    if os.path.exists(path_db):
      os.remove(path_db)
    conn = connect(path_db)
    cur = conn.cursor()
    cur.execute(f'CREATE TABLE {table_name}({sql})')
    conn.commit()
    conn.close()

  conn_workload = connect(output_workload)
  with open(workload_json) as f:
    cur_workload = conn_workload.cursor()
    for line in f:
      row = json.loads(line)
      shash = row[0]
      base64_mod = row[1]
      cur_workload.execute(
          f'INSERT INTO workload values(\'{shash}\',\'{base64_mod}\')')
  conn_workload.commit()
  conn_workload.close()

  conn_tuning_record = connect(output_tuning_record)
  with open(tuning_record_json) as f:
    cur_tuning_record = conn_tuning_record.cursor()
    for line in f:
      row = json.loads(line)
      workload_index = row[0]
      str_trace = json.dumps(row[1][0])
      run_secs = json.dumps(row[1][1])
      target = json.dumps(row[1][2])
      args_info = json.dumps(row[1][3])
      cur_tuning_record.execute(
          f'INSERT INTO tuning_record values({workload_index},\'{str_trace}\',\'{run_secs}\',\'{target}\',\'{args_info}\')')
  conn_tuning_record.commit()
  conn_tuning_record.close()


def convert_to_sqlite3(work_dir):
  workload_json = os.path.join(work_dir, 'database_workload.json')
  tuning_record_json = os.path.join(work_dir, 'database_tuning_record.json')
  output_workload = os.path.join(work_dir, 'database_workload.sqlite')
  output_tuning_record = os.path.join(
      work_dir, 'database_tuning_record.sqlite')
  convert_to_db(sqlite3.connect, workload_json, tuning_record_json,
                output_workload, output_tuning_record)
  print(f"Created {output_workload}")
  print(f"Created {output_tuning_record}")


if __name__ == "__main__":
  parse = argparse.ArgumentParser()
  parser = argparse.ArgumentParser("dbconvert")
  parser.add_argument("--work_dir", type=str, default="./log-resnet18")

  args = parser.parse_args()

  print(f"Converting {args.work_dir} into SQLite3")
  convert_to_sqlite3(args.work_dir)
