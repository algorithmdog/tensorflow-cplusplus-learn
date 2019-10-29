load("//tensorflow:tensorflow.bzl","tf_cc_binary")

tf_cc_binary(
    name="example",
    srcs=["example.cc"],
    deps=[
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
	name="example_train",
	srcs=["example_train.cc"],
	deps=[
	   "//tensorflow/cc:cc_ops",
	   "//tensorflow/cc:client_session",
	   "//tensorflow/core:tensorflow",
	],
)
