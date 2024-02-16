# SPDX-License-Identifier: Apache-2.0
import sqlite3
import base64
import json
import os
import ctypes

from typing import List, Dict, Optional, Callable
import tvm
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.utils import derived_object
from tvm.ir import IRModule


class SortTuningRecordByMeanRunSecs():
  kMaxMeanTime = 1e10


class SQLDatabase:
  """Database class backed by SQL.

  See also: PyDatabase
  """
  path_workload: str
  path_tuning_record: str
  module_equality: str
  tuning_records: List[TuningRecord]
  workloads: List[Workload]
  workload2idx: Dict[Workload, int]
  conn_func: Callable[[str], sqlite3.Connection]

  def fetchall(
      self,
      path_db: str,
      statement: str
  ) -> list:
    """Fetchall from the database.

    Parameters
    ----------
    path_db : str
        The path to the SQL database.
    statement : str
        SQL statement to execute.

    Returns
    -------
        All rows of a query result.
    """
    conn = self.conn_func(path_db)
    cur = conn.cursor()
    cur.execute(statement)
    records = cur.fetchall()
    conn.close()

    return records

  def init(
      self,
      path_workload: Optional[str] = None,
      path_tuning_record: Optional[str] = None,
      *,
      work_dir: Optional[str] = None,
      module_equality: str = "structural",
      conn_func: Callable[[str], sqlite3.Connection],
      file_ext: str,
  ) -> None:
    """Initializer.

    Parameters
    ----------
    path_workload : Optional[str] = None
        The path to the workload table. If not specified,
        will be generated from `work_dir` as `$work_dir/database_workload.(sqlite|duckdb)`.
    path_tuning_record : Optional[str] = None
        The path to the tuning record table. If not specified,
        will be generated from `work_dir` as `$work_dir/database_tuning_record.(sqlite|duckdb)`.
    work_dir : Optional[str] = None
        The work directory, if specified, will be used to generate `path_tuning_record`
        and `path_workload`.
    module_equality : Optional[str] = "structural"
        A string to specify the module equality testing and hashing method.
        "structural", "ignore-ndarray" or "anchor-block" are all the same.
    conn_func : Union[sqlite3.Connection, duckdb.DuckDBPyConnection]
        Function object for database connection.
    file_ext : str
        The extension string.
    """
    if work_dir is not None:
      if path_workload is None:
        path_workload = os.path.join(work_dir, f"database_workload.{file_ext}")
      else:
        path_workload = os.path.join(work_dir, path_workload)
      if path_tuning_record is None:
        path_tuning_record = os.path.join(
            work_dir, f"database_tuning_record.{file_ext}")
      else:
        path_tuning_record = os.path.join(work_dir, path_tuning_record)
      os.makedirs(work_dir, exist_ok=True)
    if path_workload is None:
      raise ValueError("`path_workload` is not specified.")
    if path_tuning_record is None:
      raise ValueError("`path_tuning_record` is not specified.")

    self.path_workload = path_workload
    self.path_tuning_record = path_tuning_record
    self.workloads = []
    self.workload2idx = {}
    self.module_equality = module_equality
    self.conn_func = conn_func

    if os.path.isfile(self.path_workload):
      db_workload = self.fetchall(self.path_workload, 'SELECT * FROM workload')
      for item in db_workload:
        byte_mod = base64.b64decode(item[1])
        json_mod = byte_mod[8:].decode('utf-8')
        mod = tvm.ir.load_json(json_mod)
        _ = self.commit_workload(mod, False)
    else:
      self.create_table(self.path_workload, "workload", 'shash text, mod text')

    if not os.path.isfile(self.path_tuning_record):
      self.create_table(
          self.path_tuning_record,
          "tuning_record",
          'workload_index int, trace text, run_secs text, target text, args_info text'
      )

  def db_to_tuning_record(self, item):
    workload = self.workloads[item[0]]
    trace = item[1]
    run_sec = str(item[2])
    target = item[3]
    args_info = item[4]
    str_record = '['+trace+','+run_sec+','+target+','+args_info+']'
    obj_record = json.loads(str_record)
    record = TuningRecord.from_json(obj_record, workload)
    return record

  def create_table(self, path_db: str, table: str, sql: str) -> None:
    """Create SQL table.

    Parameters
    ----------
    path_db : str
        The path to the SQL database.
    table : str
        Table name.
    sql : str
        The SQL string.
    """
    conn = self.conn_func(path_db)
    cur = conn.cursor()
    cur.execute(f'CREATE TABLE {table}({sql})')
    conn.commit()
    conn.close()

  def has_workload(self, mod: IRModule) -> bool:
    for workload in self.workloads:
      if tvm.ir.structural_equal(mod, workload.mod):
        return True
    return False

  def commit_workload(self, mod: IRModule, commit_sql: bool) -> Workload:
    if self.has_workload(mod):
      for workload in self.workloads:
        if tvm.ir.structural_equal(mod, workload.mod):
          return workload
    workload = Workload(mod)
    self.workloads.append(workload)
    self.workload2idx[workload] = len(self.workloads) - 1

    if commit_sql is True:
      self.commit_workload_to_sql(mod)

    return workload

  def commit_workload_to_sql(self, mod: IRModule) -> None:
    json_mod = tvm.ir.save_json(mod)
    json_mod_size = len(json_mod)
    byte_mod = json_mod_size.to_bytes(8, 'little') + json_mod.encode()
    base64_mod = base64.b64encode(byte_mod).decode()
    signed_shash = tvm.ir.structural_hash(mod)
    unsigned_shash = ctypes.c_uint64(signed_shash).value

    conn = self.conn_func(self.path_workload)
    cur = conn.cursor()
    cur.execute(
        f'INSERT INTO workload values(\'{unsigned_shash}\',\'{base64_mod}\')')
    conn.commit()
    conn.close()

  def commit_tuning_record(self, record: TuningRecord) -> None:
    workload_index = self.workload2idx[record.workload]
    trace = tvm.tir.schedule.Trace.as_json(record.trace)
    str_trace = json.dumps(trace)
    if record.run_secs is not None:
      run_secs = [v.value for v in record.run_secs]
    else:
      run_secs = None
    target = record.target.export()
    args_info = [x.as_json() for x in record.args_info]
    args_info = json.dumps(args_info)

    conn = self.conn_func(self.path_tuning_record)
    cur = conn.cursor()
    cur.execute(
        f'INSERT INTO tuning_record values({workload_index},\'{str_trace}\',\'{run_secs}\',\'{target}\',\'{args_info}\')')
    conn.commit()
    conn.close()

  def is_valid(self, item) -> bool:
    """Is tuning record item is valid?

    Parameters
    ----------
    db_record : record item fetched from database
        The tuning record node.

    Returns
    -------
        True(valid) or False(invalid)

    """
    run_secs = json.loads(item[2])
    if run_secs is None or len(run_secs) == 0:
      return False
    for run_sec in run_secs:
      # kMaxMeanTime(1e10) is used as a stub for undefined measurement times.
      if run_sec != SortTuningRecordByMeanRunSecs.kMaxMeanTime:
        return True
    return False

  def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
    workload_idx = self.workload2idx[workload]
    db_tuning_records = self.fetchall(
        self.path_tuning_record, f"SELECT * from tuning_record WHERE workload_index={workload_idx}")

    # remove invalid record from db_tuning_records
    db_tuning_records = [x for x in db_tuning_records if self.is_valid(x)]

    top_k_db_tuning_records = sorted(
        db_tuning_records,
        key=lambda item: sum(json.loads(
            item[2])) / len(json.loads(item[2])) if item[2] else 1e9,
    )[:top_k]

    return [self.db_to_tuning_record(item) for item in top_k_db_tuning_records]

  def get_all_tuning_records(self) -> List[TuningRecord]:
    db_tuning_records = self.fetchall(
        self.path_tuning_record, f"SELECT * from tuning_record")
    records = [self.db_to_tuning_record(item) for item in db_tuning_records]
    return records

  def __len__(self) -> int:
    return len(self.fetchall(self.path_tuning_record, f"SELECT workload_index from tuning_record"))


@derived_object
class SQLiteDatabase(PyDatabase):
  spldb: SQLDatabase

  def __init__(
      self,
      path_workload: Optional[str] = None,
      path_tuning_record: Optional[str] = None,
      *,
      work_dir: Optional[str] = None,
  ) -> None:
    self.sqldb = SQLDatabase()
    self.sqldb.init(
        path_workload,
        path_tuning_record,
        work_dir=work_dir,
        conn_func=lambda x: sqlite3.connect(x),
        file_ext="sqlite")

  def has_workload(self, mod: IRModule) -> bool:
    return self.sqldb.has_workload(mod)

  def commit_workload(self, mod: IRModule) -> Workload:
    return self.sqldb.commit_workload(mod, True)

  def commit_tuning_record(self, record: TuningRecord) -> None:
    self.sqldb.commit_tuning_record(record)

  def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
    return self.sqldb.get_top_k(workload, top_k)

  def get_all_tuning_records(self) -> List[TuningRecord]:
    return self.sqldb.get_all_tuning_records()

  def __len__(self) -> int:
    return len(self.sqldb)
