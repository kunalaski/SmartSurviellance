	??{??]?@??{??]?@!??{??]?@	[??u????[??u????![??u????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??{??]?@xz?,C?@A$(~?u[?@Yc?ZB>h@*	????&)bA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator0L?
V??@!_1Dx?X@)0L?
V??@1_1Dx?X@:Preprocessing2F
Iterator::Model?3??7???!?s?Z?v?)?~j?t???1#wK^?Hv?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap???Kw??@!??????X@)?-????1?v2;Zf?:Preprocessing2P
Iterator::Model::Prefetch?(??0??!?߉?? ?)?(??0??1?߉?? ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9[??u????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	xz?,C?@xz?,C?@!xz?,C?@      ??!       "      ??!       *      ??!       2	$(~?u[?@$(~?u[?@!$(~?u[?@:      ??!       B      ??!       J	c?ZB>h@c?ZB>h@!c?ZB>h@R      ??!       Z	c?ZB>h@c?ZB>h@!c?ZB>h@JCPU_ONLYY[??u????b 