	??Q?N?@??Q?N?@!??Q?N?@	??"?q?????"?q???!??"?q???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q?N?@??H@A?h o?K?@Y=,Ԛ??@*	???)??fA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator(??'?@!?d???X@)(??'?@1?d???X@:Preprocessing2F
Iterator::Model?A`??"??!?ys??s??)?ǘ?????1?i?]Q???:Preprocessing2P
Iterator::Model::Prefetchk?w??#??!???#K?)k?w??#??1???#K?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap7?[?'?@!d<?c?X@)?{??Pk??1:?}???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??"?q???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??H@??H@!??H@      ??!       "      ??!       *      ??!       2	?h o?K?@?h o?K?@!?h o?K?@:      ??!       B      ??!       J	=,Ԛ??@=,Ԛ??@!=,Ԛ??@R      ??!       Z	=,Ԛ??@=,Ԛ??@!=,Ԛ??@JCPU_ONLYY??"?q???b 