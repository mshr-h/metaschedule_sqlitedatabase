# SPDX-License-Identifier: Apache-2.0
from sql_database import SQLiteDatabase

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

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

# prepare sample image
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
my_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img = my_preprocess(img)
img = np.expand_dims(img.numpy(), 0)

# MetaSchedule
opt_level = 3
target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 4")
dev = tvm.cpu(0)
dtype = "float32"
sqldb = SQLiteDatabase(work_dir=f"./log-{model_name}/")

# compile w/o MetaScheduler
with tvm.transform.PassContext(opt_level=opt_level):
  lib = relay.build(mod, target=target, params=params)

# compile w/ MetaScheduler
lib_ms = ms.relay_integration.compile_relay(
    sqldb, mod, target, params, opt_level=opt_level)


# run compiled module
def run_module(lib):
  module = graph_executor.GraphModule(lib["default"](dev))
  module.set_input(input_name, tvm.nd.array(img.astype(dtype)))
  module.run()
  return module.get_output(0).numpy()


output_tvm = run_module(lib)
top1_tvm = np.argmax(output_tvm)

output_tvm_ms = run_module(lib_ms)
top1_tvm_ms = np.argmax(output_tvm_ms)

# PyTorch
with torch.no_grad():
  img_torch = torch.from_numpy(img)
  output_torch = model(img_torch).numpy()
  top1_torch = np.argmax(output_torch)

print(f"top1 tvm w/o MetaScheduler: {top1_tvm}, {output_tvm[0][top1_tvm]}")
print(
    f"top1 tvm w/ MetaScheduler : {top1_tvm_ms}, {output_tvm_ms[0][top1_tvm_ms]}")
print(
    f"top1 torch                : {top1_torch}, {output_torch[0][top1_torch]}")
np.testing.assert_allclose(output_torch, output_tvm, atol=1e-4, rtol=1e-4)
np.testing.assert_allclose(output_torch, output_tvm_ms, atol=1e-4, rtol=1e-4)
np.testing.assert_allclose(output_tvm, output_tvm_ms, atol=1e-4, rtol=1e-4)
