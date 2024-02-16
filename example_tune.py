# SPDX-License-Identifier: Apache-2.0
from sql_database import SQLiteDatabase

import torch
import tvm
from tvm import relay
from tvm import meta_schedule as ms

# prepare resnet18 model
model_name = "resnet18"
input_name = "input0"
input_shape = (1, 3, 224, 224)
model = torch.hub.load("pytorch/vision:v0.15.2",
                       model_name, weights="DEFAULT").eval()
input_data = torch.rand(input_shape, dtype=torch.float32)
script_module = torch.jit.trace(model, input_data).eval()
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

# MetaSchedule
opt_level = 3
target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 4")
dev = tvm.cpu(0)
dtype = "float32"
max_trials_global = 500
num_trials_per_iter = 10
work_dir = f"./log-{model_name}/"
sqldb = SQLiteDatabase(work_dir=work_dir)
sqldb = ms.relay_integration.tune_relay(mod,
                                        params,
                                        target,
                                        work_dir,
                                        database=sqldb,
                                        max_trials_global=max_trials_global,
                                        num_trials_per_iter=num_trials_per_iter)
