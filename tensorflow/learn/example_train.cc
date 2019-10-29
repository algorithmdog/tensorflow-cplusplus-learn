/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using tensorflow::string;
using tensorflow::int32;

namespace tensorflow{
namespace example{

int run() {

  Scope scope = Scope::NewRootScope();


  using namespace ::tensorflow::ops; 
  auto x = Placeholder(scope, DT_FLOAT);
  auto y = Placeholder(scope, DT_FLOAT);

  // weights init
  auto w1 = Variable(scope, {3, 3}, DT_FLOAT);
  auto assign_w1 = Assign(scope, w1, RandomNormal(scope, {3, 3}, DT_FLOAT));

  auto w2 = Variable(scope, {3, 2}, DT_FLOAT);
  auto assign_w2 = Assign(scope, w2, RandomNormal(scope, {3, 2}, DT_FLOAT));

  auto w3 = Variable(scope, {2, 1}, DT_FLOAT);
  auto assign_w3 = Assign(scope, w3, RandomNormal(scope, {2, 1}, DT_FLOAT));

  // bias init
  auto b1 = Variable(scope, {1, 3}, DT_FLOAT);
  auto assign_b1 = Assign(scope, b1, RandomNormal(scope, {1, 3}, DT_FLOAT));

  auto b2 = Variable(scope, {1, 2}, DT_FLOAT);
  auto assign_b2 = Assign(scope, b2, RandomNormal(scope, {1, 2}, DT_FLOAT));

  auto b3 = Variable(scope, {1, 1}, DT_FLOAT);
  auto assign_b3 = Assign(scope, b3, RandomNormal(scope, {1, 1}, DT_FLOAT));

  // layers
  auto layer_1 = Tanh(scope, Tanh(scope, Add(scope, MatMul(scope, x, w1), b1)));
  auto layer_2 = Tanh(scope, Add(scope, MatMul(scope, layer_1, w2), b2));
  auto layer_3 = Tanh(scope, Add(scope, MatMul(scope, layer_2, w3), b3));


  // loss calculation
  auto loss = ReduceMean(scope, Square(scope, Sub(scope, layer_3, y)), {0, 1});
                  

  // add the gradients operations to the graph
  std::vector<Output> grad_outputs;
  TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));

  // update the weights and bias using gradient descent
  auto apply_w1 = ApplyGradientDescent(scope, w1, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[0]});
  auto apply_w2 = ApplyGradientDescent(scope, w2, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[1]});
  auto apply_w3 = ApplyGradientDescent(scope, w3, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[2]});
  auto apply_b1 = ApplyGradientDescent(scope, b1, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[3]});
  auto apply_b2 = ApplyGradientDescent(scope, b2, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[4]});
  auto apply_b3 = ApplyGradientDescent(scope, b3, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[5]});

  ClientSession session(scope);
  std::vector<Tensor> outputs;

  // init the weights and biases by running the assigns nodes once
  TF_CHECK_OK(session.Run({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));

  // training steps
  for (int i = 0; i < 5000; ++i) {
    if (i % 100 == 0) {
      TF_CHECK_OK(session.Run({{x, 1.0}, {y, 2.0}}, {loss}, &outputs));
      std::cout << "Loss after " << i << " steps " << outputs[0].scalar<float>() << std::endl;
    }
    // nullptr because the output from the run is useless
    TF_CHECK_OK(session.Run({{x, 1.0}, {y, 2.0}}, {apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3}, nullptr));
  }
}
}
}

int main(int argc, char* argv[]) {
  tensorflow::example::run();
}
